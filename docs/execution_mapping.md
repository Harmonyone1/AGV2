# Execution Mapping Guide

This environment mirrors the TradeLocker order flow. Use these fields from `info["execution"]` when wiring PPO decisions into the live broker:

- `requested_position`: signed target size from the policy before session/limit constraints.
- `target_position`: size after session and limit-fill evaluation (what the sim actually tries to hold).
- `order_type`: `market` or `limit` (matches TradeLocker order types). Use it to choose `order_type`/`route` when building the API payload.
- `limit_offset_bps`: optional price improvement (basis points). Convert it into absolute price offsets when setting limit prices.
- `take_profit_bps` / `stop_loss_bps`: bracket distances. Translate to TP/SL prices around the post-fill entry.
- `bracket_trigger`: `take_profit`, `stop_loss`, or `None` indicating that a bracket just closed the position. When non-null, send the corresponding close order live.
- `fill_ratio`: domain-randomized fill percentage (0–1). Use it as an expectation for partial fills.
- `session_open`: `False` when the env suppressed execution due to closed sessions. Avoid submitting new orders in those windows.
- `trade_cost_bps`, `spread_bps`, `slippage_bps`, `commission_bps`: cost components applied during the step, useful for validating live PnL vs. sim.

Map live actions as follows:

1. Sample PPO action → decode using the same action map (position, order_type, offsets, brackets).
2. Convert `requested_position` into a target net size; compare against current broker position to compute order quantity.
3. If `order_type == "limit"`, adjust your limit price using `limit_offset_bps` relative to the best bid/ask.
4. Attach TP/SL legs using `take_profit_bps` and `stop_loss_bps` (basis points relative to fill price).
5. Respect `session_open`; if False, delay order entry until the next open session.
6. Monitor fills; when live PnL hits the bracket thresholds, close positions and tag them with the same trigger semantics.
