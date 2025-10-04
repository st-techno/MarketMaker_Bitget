# Copyright @ ST Technologies

## Comprehensive Production-Grade Python Market-Making system for BTC-USDT spot and futures on centralized order books like Bitget that includes:

Full rebalancing logic with position limits and inventory hedging

Hedging between spot and futures to reduce directional risk

Volatility prediction using GARCH for dynamic sizing

Latency measurement for trading performance monitoring

Real-time interactive dashboard with operational KPIs

Modular, thread-safe architecture with robust logging

This design mimics real market scenarios and institutional best practices for risk control and performance.

## Explanation of Rebalancing and Hedging Logic:

The market maker continuously monitors spot and futures inventories.

If spot inventory exceeds max allowed (risk param), it places limit orders to reduce exposure.

The hedge logic calculates a hedge ratio to offset spot inventory by futures contracts dynamically.

The hedge order size is clamped within predefined futures max inventory boundaries.

Order sizing uses GARCH volatility forecast (inverse scaling) to adjust aggressiveness.

Latency is continuously monitored to ensure API responsiveness.

A real-time dashboard visualizes KPIs including spreads, inventories, volatility forecasts, and latency.

Thread-safe state management ensures accurate KPI and inventory tracking in production.

This structure reflects a realistic production deployment with risk controls, position hedging, dynamic sizing, and live operational telemetry for institutional crypto market makers.

