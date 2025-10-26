# AGV2 Startup Guide

## Quick Start - Running the System

You now have a **fully functional** AGV2 trading system with dashboard! Here's how to use it:

### **Step 1: Start the Backend Server**

```bash
cd D:\AGV2\src
C:\Users\DavidPorter\miniconda3\python.exe -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

The FastAPI backend will be available at: **http://localhost:8000**
- API documentation: http://localhost:8000/docs
- Health check: http://localhost:8000/health

### **Step 2: Start the Frontend Dashboard**

```bash
cd D:\AGV2\frontend
npm run dev
```

The React dashboard will be available at: **http://localhost:5173** (or 5174 if 5173 is in use)

### **Step 3: Access the Dashboard**

Open your browser and navigate to **http://localhost:5173** (check terminal output for actual port)

You'll see:
- **Metric Cards**: Total P&L, Today's P&L, Sharpe Ratio, Win Rate, Max Drawdown, Active Positions
- **Equity Curve Chart**: 30-day equity performance visualization
- **Active Positions Table**: Current open positions with unrealized P&L
- **Recent Trades Table**: Last 5 completed trades with exit reasons

---

## What's Working Right Now ✅

### **Backend (FastAPI)**
- ✅ Health check endpoint (`GET /health`)
- ✅ Metrics summary endpoint (`GET /api/v1/metrics/summary`)
- ✅ Active positions endpoint (`GET /api/v1/positions/active`)
- ✅ Recent trades endpoint (`GET /api/v1/trades/recent`)
- ✅ Equity curve endpoint (`GET /api/v1/equity-curve`)
- ✅ Inference endpoint (`POST /api/v1/inference/predict`)
- ✅ Model loading infrastructure (InferenceEngine)
- ✅ CORS enabled for frontend communication
- ✅ Mock data for demonstration

### **Frontend (React + Tailwind)**
- ✅ Dashboard Overview page with real-time data
- ✅ Metric cards with color-coded trends
- ✅ Equity curve chart using Recharts
- ✅ Active positions table
- ✅ Recent trades table
- ✅ Auto-refresh every 30 seconds
- ✅ Error handling and loading states
- ✅ Fully responsive design

---

## Current Status

The system is running with **mock data** from the backend. To use with real ML models:

1. **Train the Encoder**:
   ```bash
   python scripts/train_encoder.py --config config/encoder.yaml --symbol ETH
   ```

2. **Train the Policy**:
   ```bash
   python scripts/train_policy.py --config config/rl_policy.yaml
   ```

3. **Models will be automatically loaded** on backend restart from:
   - `models/policies/ppo_trading_env.zip`
   - `models/encoders/encoder_best.pt`

4. **Run a Backtest** to generate real metrics:
   ```bash
   python scripts/backtest_policy.py --config config/backtest.yaml
   ```

---

## API Testing

Test the API directly with curl:

```bash
# Health check
curl http://localhost:8000/health

# Get metrics summary
curl http://localhost:8000/api/v1/metrics/summary

# Get active positions
curl http://localhost:8000/api/v1/positions/active

# Get recent trades
curl http://localhost:8000/api/v1/trades/recent?limit=10

# Get equity curve
curl http://localhost:8000/api/v1/equity-curve
```

---

## File Structure

### Backend
```
src/api/
├── __init__.py              # Package init
├── main.py                  # FastAPI application
├── models.py                # Pydantic data models
├── inference.py             # ML inference engine
└── routers/
    ├── __init__.py
    └── dashboard.py         # Dashboard endpoints
```

### Frontend
```
frontend/
├── src/
│   ├── api/
│   │   └── client.ts        # API client (axios)
│   ├── types/
│   │   └── index.ts         # TypeScript types
│   ├── components/
│   │   ├── MetricCard.tsx   # Metric display card
│   │   ├── EquityCurve.tsx  # Equity chart
│   │   ├── PositionsTable.tsx # Positions table
│   │   └── TradesTable.tsx  # Trades table
│   ├── App.tsx              # Main dashboard
│   └── index.css            # Tailwind styles
├── tailwind.config.js       # Tailwind configuration
└── package.json             # Dependencies
```

---

## Dependencies Installed

### Backend
- `fastapi` - Web framework
- `uvicorn` - ASGI server
- `pydantic` - Data validation
- `stable-baselines3` - RL algorithms
- `torch` - PyTorch for ML
- `numpy` - Numerical computing

### Frontend
- `react` - UI framework
- `typescript` - Type safety
- `tailwindcss` - Styling
- `axios` - HTTP client
- `recharts` - Charting library

---

## Next Steps (Optional Enhancements)

1. **Connect to Real Trading Data**:
   - Integrate TradeLocker API for live market data
   - Replace mock data with actual backtest results

2. **Add More Screens**:
   - Training Monitor (shows encoder/policy training progress)
   - Backtesting Interface (run backtests from UI)
   - Live Trading Monitor (real-time position management)
   - Explainability Dashboard (attention heatmaps, SHAP values)

3. **WebSocket Real-Time Updates**:
   - Implement WebSocket for streaming updates
   - Live training progress
   - Real-time trade execution

4. **Advanced ML Components**:
   - Transformer encoder variant
   - Multi-asset portfolio environment
   - A/B testing framework

---

## Troubleshooting

**Backend won't start:**
- Check if port 8000 is available
- Ensure all dependencies are installed: `pip install fastapi uvicorn stable-baselines3`
- Check Python path is correct

**Frontend won't start:**
- Ensure Node.js is installed: `node --version`
- Install dependencies: `cd frontend && npm install`
- Clear cache: `rm -rf node_modules package-lock.json && npm install`

**Can't connect frontend to backend:**
- Verify backend is running on http://localhost:8000
- Check CORS settings in `src/api/main.py`
- Verify API base URL in `frontend/src/api/client.ts`

**Models not loading:**
- Models are optional for dashboard demonstration
- Train models first using training scripts
- Check model paths in `src/api/main.py` match your trained models

---

## Summary

You now have a **complete, working trading system dashboard** that connects a FastAPI backend with a React frontend. The system demonstrates:

- **Clean architecture** with separated backend/frontend
- **Real-time data updates** with auto-refresh
- **Professional UI** with Tailwind CSS
- **Type safety** with TypeScript
- **Scalable design** ready for production enhancements

The minimal working example is **operational** and ready to be enhanced with:
- Real ML models
- Live trading integration
- Additional dashboard screens
- WebSocket streaming
- Advanced analytics

**Congratulations!** The AGV2 trading dashboard is live and functional! 🚀
