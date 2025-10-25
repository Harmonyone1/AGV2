import numpy as np
import pytest

from backtest import VectorizedBacktester


def test_vectorized_backtester_metrics():
    returns = np.array([0.01, -0.005, 0.002], dtype=np.float32)
    signals = np.array([1.0, 1.0, -1.0], dtype=np.float32)
    bt = VectorizedBacktester(returns, trading_cost_bps=0.0)
    result = bt.run(signals)
    expected = float(np.sum(returns * signals))
    assert result.total_return == pytest.approx(expected)
    assert result.equity_curve.shape == returns.shape


def test_vectorized_backtester_shape_guard():
    bt = VectorizedBacktester([0.01, 0.02])
    with pytest.raises(ValueError):
        bt.run([1.0])
