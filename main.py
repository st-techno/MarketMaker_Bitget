import ccxt
import pandas as pd
import numpy as np
import logging
import time
import threading
from arch import arch_model
import dash
from dash import dcc, html
import plotly.graph_objs as go

# === Logging setup ===
logging.basicConfig(filename='market_maker.log',
                    level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')

# === Latency Monitor (measures API response delays) ===
class LatencyMonitor:
    def __init__(self):
        self.latencies = []

    def ping_exchange(self, exchange, symbol):
        try:
            start = time.time()
            exchange.fetch_ticker(symbol)
            latency = time.time() - start
            self.latencies.append(latency)
            return latency
        except Exception as e:
            logging.error(f"Latency ping error: {e}")
            return None

    def avg_latency(self):
        if not self.latencies:
            return 0.0
        return np.mean(self.latencies[-100:])

# === Volatility Predictor using GARCH ===
class VolatilityPredictor:
    def __init__(self):
        self.model = None
        self.latest_vol = None

    def fit(self, returns: pd.Series):
        self.model = arch_model(returns, mean='zero', vol='GARCH', p=1, o=0, q=1).fit(disp='off')
        logging.info("GARCH model fitted.")
    
    def predict_volatility(self, horizon=1) -> float:
        if not self.model:
            raise RuntimeError("Volatility model not fit.")
        forecast = self.model.forecast(horizon=horizon)
        vol = np.sqrt(forecast.variance.iloc[-1].values[0])
        self.latest_vol = vol
        return vol

# === Market Maker Core ===
class MarketMaker:
    def __init__(self, exchange, spot_symbol, futures_symbol, risk_params, vol_predictor, latency_monitor):
        self.exchange = exchange
        self.spot = spot_symbol
        self.futures = futures_symbol
        self.risk = risk_params
        self.vol_predictor = vol_predictor
        self.latency_monitor = latency_monitor
        
        # Internal state
        self.inventory_spot = 0.0
        self.inventory_futures = 0.0
        self.pnl = 0.0
        
        # KPIs for dashboard and monitoring
        self.kpi = {
            'latency': [],
            'volatility': [],
            'spread': [],
            'net_inventory': [],
            'hedge_ratio': [],
            'trades_executed': 0,
            'trade_volume': 0,
            'pnl': []
        }
        self.lock = threading.Lock()
    
    # Fetch order book bids and asks
    def fetch_order_book(self, symbol):
        try:
            ob = self.exchange.fetch_order_book(symbol)
            return ob['bids'], ob['asks']
        except Exception as e:
            logging.error(f"Failed fetch order book {symbol}: {e}")
            return [], []
    
    # Calculate spread from top bids and asks
    def calc_spread(self, bids, asks):
        if not bids or not asks:  # No valid data
            return None
        spread = asks[0][0] - bids[0][0]
        self.kpi['spread'].append(spread)
        return spread
    
    # Place an order with error handling
    def place_limit_order(self, symbol, side, price, amount):
        try:
            order = self.exchange.create_limit_order(symbol, side, amount, price)
            logging.info(f"Order placed: {side} {symbol} {amount}@{price}")
            with self.lock:
                self.kpi['trades_executed'] += 1
                self.kpi['trade_volume'] += amount
            return order
        except Exception as e:
            logging.error(f"Order placement failed: {e}")
            return None
    
    # Calculate hedge ratio based on volatility and current exposures
    def calc_hedge_ratio(self):
        # Simplistic hedge ratio: match spot inventory with futures
        # In production, could use beta from regression or GARCH conditional covariances
        try:
            ratio = -self.inventory_spot / (self.inventory_futures if self.inventory_futures != 0 else 1)
        except ZeroDivisionError:
            ratio = 0.0
        self.kpi['hedge_ratio'].append(ratio)
        return ratio
    
    # Execute rebalancing to maintain target inventory levels and hedge
    def rebalance_positions(self):
        with self.lock:
            net_inventory = self.inventory_spot + self.inventory_futures
            self.kpi['net_inventory'].append(net_inventory)
            # Risk limits checks
            if abs(self.inventory_spot) > self.risk['max_spot_inventory']:
                # Generate limit orders to offload excess spot position
                logging.warning(f"Rebalancing spot inventory: {self.inventory_spot}")
                # Example logic: market sell or buy to reduce exposure:
                side = 'sell' if self.inventory_spot > 0 else 'buy'
                qty = abs(self.inventory_spot) - self.risk['max_spot_inventory']
                best_price = self.fetch_order_book(self.spot)[0][0] if side == 'sell' else self.fetch_order_book(self.spot)[1][0]
                self.place_limit_order(self.spot, side, best_price, qty)
                self.inventory_spot -= qty if side == 'sell' else -qty
            
            # Hedge spot positions using futures contracts
            hedge_qty = -self.inventory_spot  # Hedge full spot exposure initially
            # Limit hedge size to max futures inventory limit
            hedge_qty = max(min(hedge_qty, self.risk['max_futures_inventory'] - self.inventory_futures), 
                            -self.risk['max_futures_inventory'] - self.inventory_futures)
            
            if abs(hedge_qty) > self.risk['min_hedge_size']:
                # Place limit futures hedge order at best price
                side = 'sell' if hedge_qty < 0 else 'buy'
                price = self.fetch_order_book(self.futures)[0][0] if side == 'sell' else self.fetch_order_book(self.futures)[1][0]
                self.place_limit_order(self.futures, side, price, abs(hedge_qty))
                self.inventory_futures += hedge_qty
    
    # Basic risk management: inventory and position checks
    def risk_management(self):
        if abs(self.inventory_spot) > self.risk['max_spot_inventory'] or abs(self.inventory_futures) > self.risk['max_futures_inventory']:
            logging.warning("Inventory risk limits breached. Initiating rebalancing.")
            self.rebalance_positions()
    
    # Market making quoting and execution logic
    def market_make(self, returns):
        # Fetch order books
        bids_spot, asks_spot = self.fetch_order_book(self.spot)
        bids_fut, asks_fut = self.fetch_order_book(self.futures)
        if not bids_spot or not asks_spot or not bids_fut or not asks_fut:
            return  # Skip iteration if data incomplete
        
        # Calculate spreads
        spread_spot = self.calc_spread(bids_spot, asks_spot)
        spread_fut = self.calc_spread(bids_fut, asks_fut)
        
        # Measure latency and record
        latency = self.latency_monitor.ping_exchange(self.exchange, self.spot)
        if latency is not None:
            self.kpi['latency'].append(latency)
        
        # Predict volatility (GARCH)
        vol = self.vol_predictor.predict_volatility()
        self.kpi['volatility'].append(vol)
        
        # Dynamic order size scaled to volatility (inverse)
        base_order_size = self.risk['base_order_size']
        vol_adj_size = max(self.risk['min_order_size'], base_order_size / (vol * 100 + 1e-8))  # scaled
        
        mid_spot = (bids_spot[0][0] + asks_spot[0][0]) / 2
        bid_price = mid_spot - spread_spot / 4
        ask_price = mid_spot + spread_spot / 4
        
        # Place bid and ask orders on spot market to capture spread
        buy_order = self.place_limit_order(self.spot, 'buy', bid_price, vol_adj_size)
        sell_order = self.place_limit_order(self.spot, 'sell', ask_price, vol_adj_size)
        
        # Update spot inventory estimation
        # For this simplified example, we simulate fill and inventory changes:
        # In real production, use websocket or order fill callbacks
        self.inventory_spot += vol_adj_size - vol_adj_size  # Net zero here, update after fills
        
        # Rebalance and hedge inventory positions dynamically
        self.risk_management()
    
    # Main loop running market making logic
    def run(self, returns):
        while True:
            try:
                self.market_make(returns)
                time.sleep(self.risk['update_interval'])
            except Exception as e:
                logging.error(f"Error in main loop: {e}")

# === Real-time Dashboard ===
def run_dashboard(market_maker):
    app = dash.Dash(__name__)

    app.layout = html.Div([
        html.H2("BTC-USDT Market Maker - Live KPIs"),
        dcc.Graph(id='live-graph'),
        dcc.Interval(id='interval', interval=1000, n_intervals=0)
    ])

    @app.callback(
        dash.dependencies.Output('live-graph', 'figure'),
        [dash.dependencies.Input('interval', 'n_intervals')]
    )
    def update_graph(_):
        with market_maker.lock:
            fig = go.Figure()
            fig.add_trace(go.Scatter(y=market_maker.kpi['latency'], mode='lines+markers', name='Latency (s)'))
            fig.add_trace(go.Scatter(y=market_maker.kpi['volatility'], mode='lines+markers', name='Predicted Volatility'))
            fig.add_trace(go.Scatter(y=market_maker.kpi['spread'], mode='lines+markers', name='Bid-Ask Spread'))
            fig.add_trace(go.Scatter(y=market_maker.kpi['net_inventory'], mode='lines+markers', name='Net Inventory'))
            fig.add_trace(go.Scatter(y=market_maker.kpi['hedge_ratio'], mode='lines+markers', name='Hedge Ratio'))
            fig.update_layout(title='Market Making KPIs', xaxis_title='Time (sec intervals)', yaxis_title='Value')
            return fig

    app.run_server(debug=True, use_reloader=False)

# === Main execution ===
if __name__ == "__main__":
    # Set up exchange with credentials
    exchange = ccxt.bitget({
        'apiKey': 'YOUR_API_KEY',
        'secret': 'YOUR_SECRET',
        'enableRateLimit': True
    })

    spot_symbol = "BTC/USDT"
    futures_symbol = "BTCUSDT_PERP"

    # Load historical price data (close) for volatility model fitting
    df = pd.read_csv("btc_usdt_tick_data.csv")
    prices = df['close']
    returns = 100 * np.log(prices / prices.shift(1))
    returns.dropna(inplace=True)

    # Instantiate components
    vol_predictor = VolatilityPredictor()
    vol_predictor.fit(returns)

    latency_monitor = LatencyMonitor()

    risk_params = {
        'max_spot_inventory': 5.0,       # Max allowed spot coins held
        'max_futures_inventory': 10.0,   # Max futures contracts held (hedging)
        'min_hedge_size': 0.01,          # Minimum size to hedge positions
        'base_order_size': 0.1,          # Base order size for quoting
        'min_order_size': 0.01,          # Minimum order size for exchange constraints
        'update_interval': 1.0            # Market making loop frequency in seconds
    }

    mm = MarketMaker(exchange, spot_symbol, futures_symbol, risk_params, vol_predictor, latency_monitor)

    # Start market making loop in background thread
    threading.Thread(target=mm.run, args=(returns,), daemon=True).start()

    # Run dashboard (blocking UI thread)
    run_dashboard(mm)
