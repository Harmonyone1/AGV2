/**
 * AGV2 Trading Dashboard - Main Application
 */
import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import api from './api/client';
import type { MetricsSummary, Position, Trade, EquityCurvePoint } from './types';
import MetricCard from './components/MetricCard';
import EquityCurve from './components/EquityCurve';
import PositionsTable from './components/PositionsTable';
import TradesTable from './components/TradesTable';
import SystemHealthCard from './components/SystemHealthCard';
import RiskOverviewCard from './components/RiskOverviewCard';

type MetricCardConfig = {
  title: string;
  value: string | number;
  subtitle?: string;
  trend?: 'up' | 'down' | 'neutral';
};

const DEFAULT_METRICS: MetricsSummary = {
  total_pnl: 0,
  total_pnl_pct: 0,
  today_pnl: 0,
  today_pnl_pct: 0,
  sharpe_ratio: 0,
  win_rate: 0,
  max_drawdown: 0,
  max_drawdown_duration_days: 0,
  active_positions: 0,
  total_trades: 0,
  avg_trade_duration_hours: 0,
  current_equity: 0,
  initial_equity: 0,
};

const currencyFormatter = new Intl.NumberFormat('en-US', {
  style: 'currency',
  currency: 'USD',
  minimumFractionDigits: 2,
  maximumFractionDigits: 2,
});

const formatCurrency = (value: number) => currencyFormatter.format(value ?? 0);
const formatPercent = (value: number, fractionDigits = 2) => `${value.toFixed(fractionDigits)}%`;

function App() {
  const [metrics, setMetrics] = useState<MetricsSummary | null>(null);
  const [positions, setPositions] = useState<Position[]>([]);
  const [trades, setTrades] = useState<Trade[]>([]);
  const [equityCurve, setEquityCurve] = useState<EquityCurvePoint[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [lastUpdated, setLastUpdated] = useState<Date | null>(null);
  const isMountedRef = useRef(true);

  useEffect(() => {
    return () => {
      isMountedRef.current = false;
    };
  }, []);

  const fetchDashboard = useCallback(async () => {
    try {
      setLoading(true);
      setError(null);

      const [metricsData, positionsData, tradesData, equityData] = await Promise.all([
        api.getMetricsSummary(),
        api.getActivePositions(),
        api.getRecentTrades(8),
        api.getEquityCurve(),
      ]);

      if (!isMountedRef.current) {
        return;
      }

      setMetrics(metricsData);
      setPositions(positionsData);
      setTrades(tradesData);
      setEquityCurve(equityData);
      setLastUpdated(new Date());
    } catch (err) {
      if (!isMountedRef.current) {
        return;
      }
      setError(err instanceof Error ? err.message : 'Failed to fetch data');
      console.error('Error fetching data:', err);
    } finally {
      if (isMountedRef.current) {
        setLoading(false);
      }
    }
  }, []);

  useEffect(() => {
    fetchDashboard();
    const interval = setInterval(fetchDashboard, 30000);
    return () => clearInterval(interval);
  }, [fetchDashboard]);

  const summary = metrics ?? DEFAULT_METRICS;
  const metricsReady = Boolean(metrics);

  const metricCards = useMemo<MetricCardConfig[]>(
    () => [
      {
        title: 'Total P&L',
        value: formatCurrency(summary.total_pnl),
        subtitle: metricsReady ? formatPercent(summary.total_pnl_pct) : 'Awaiting data',
        trend: summary.total_pnl >= 0 ? 'up' : 'down',
      },
      {
        title: "Today's P&L",
        value: formatCurrency(summary.today_pnl),
        subtitle: metricsReady ? formatPercent(summary.today_pnl_pct) : 'Awaiting data',
        trend: summary.today_pnl >= 0 ? 'up' : 'down',
      },
      {
        title: 'Sharpe Ratio',
        value: metricsReady ? summary.sharpe_ratio.toFixed(2) : '--',
        subtitle: metricsReady ? 'Risk-adjusted return' : 'Awaiting data',
        trend: 'neutral',
      },
      {
        title: 'Win Rate',
        value: metricsReady ? formatPercent(summary.win_rate * 100, 1) : '--',
        subtitle: 'Across closed trades',
        trend: 'neutral',
      },
      {
        title: 'Max Drawdown',
        value: metricsReady ? formatPercent(summary.max_drawdown) : '--',
        subtitle: metricsReady
          ? `${summary.max_drawdown_duration_days.toFixed(0)} day duration`
          : 'Awaiting data',
        trend: 'down',
      },
      {
        title: 'Active Positions',
        value: summary.active_positions,
        subtitle: metricsReady ? `${summary.total_trades} total trades` : 'Awaiting data',
        trend: 'neutral',
      },
    ],
    [metricsReady, summary],
  );

  const heroQuickStats = useMemo(
    () => [
      { label: 'Total trades', value: summary.total_trades },
      { label: 'Avg trade duration', value: `${summary.avg_trade_duration_hours.toFixed(1)}h` },
      { label: 'Drawdown duration', value: `${summary.max_drawdown_duration_days.toFixed(0)}d` },
    ],
    [summary],
  );

  const initialLoad = loading && !metricsReady && positions.length === 0 && trades.length === 0;

  if (initialLoad) {
    return (
      <div className="flex min-h-screen items-center justify-center bg-slate-950">
        <div className="rounded-2xl border border-slate-800 bg-slate-900 px-8 py-6 text-center text-slate-200 shadow-2xl">
          <p className="text-sm font-semibold uppercase tracking-[0.3em] text-indigo-400">AGV2</p>
          <p className="mt-4 text-lg font-semibold">Bootstrapping live telemetry...</p>
          <p className="mt-2 text-sm text-slate-400">Starting data feeds and compiling UI.</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="flex min-h-screen items-center justify-center bg-slate-950 px-4">
        <div className="w-full max-w-lg rounded-2xl border border-rose-200 bg-rose-50 px-6 py-5 text-rose-800 shadow">
          <p className="text-sm font-semibold uppercase tracking-[0.3em]">Data Error</p>
          <p className="mt-2 text-lg font-semibold">Unable to load dashboard</p>
          <p className="mt-1 text-sm">{error}</p>
          <div className="mt-4 flex gap-3">
            <button
              type="button"
              className="rounded-full bg-rose-600 px-4 py-2 text-sm font-semibold text-white shadow"
              onClick={fetchDashboard}
            >
              Retry
            </button>
            <button
              type="button"
              className="rounded-full border border-rose-200 px-4 py-2 text-sm font-semibold text-rose-700"
              onClick={() => setError(null)}
            >
              Dismiss
            </button>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-slate-950 text-slate-50">
      <header className="border-b border-slate-800 bg-gradient-to-br from-slate-900 to-slate-950">
        <div className="mx-auto flex max-w-7xl flex-col gap-6 px-4 py-10 sm:px-6 lg:px-8">
          <div className="flex flex-col gap-6 md:flex-row md:items-end md:justify-between">
            <div>
              <p className="text-sm font-semibold uppercase tracking-[0.4em] text-indigo-300">AGV2</p>
              <h1 className="mt-3 text-4xl font-semibold text-white">Trading Dashboard</h1>
              <p className="mt-2 max-w-2xl text-base text-slate-300">
                Live performance, risk, and execution telemetry for the autonomous grid vault.
              </p>
            </div>
            <div className="flex flex-col gap-3 text-sm text-slate-300 sm:flex-row sm:items-center">
              <span>{lastUpdated ? `Synced ${lastUpdated.toLocaleTimeString()}` : 'Awaiting first data pull'}</span>
              <div className="flex gap-3">
                <button
                  type="button"
                  onClick={fetchDashboard}
                  className="rounded-full border border-white/30 px-4 py-2 font-semibold text-white transition hover:bg-white/10"
                >
                  Refresh data
                </button>
                <button
                  type="button"
                  className="rounded-full bg-indigo-500 px-4 py-2 font-semibold text-white shadow hover:bg-indigo-400"
                >
                  Start session
                </button>
              </div>
            </div>
          </div>
          <dl className="grid gap-4 sm:grid-cols-3">
            {heroQuickStats.map((stat) => (
              <div key={stat.label} className="rounded-2xl border border-white/10 bg-white/5 px-4 py-3">
                <dt className="text-xs font-semibold uppercase tracking-[0.3em] text-slate-300">{stat.label}</dt>
                <dd className="mt-2 text-2xl font-semibold text-white">{stat.value}</dd>
              </div>
            ))}
          </dl>
        </div>
      </header>

      <main className="mx-auto max-w-7xl px-4 py-10 sm:px-6 lg:px-8">
        <section className="grid gap-6 sm:grid-cols-2 xl:grid-cols-3">
          {metricCards.map((card) => (
            <MetricCard key={card.title} title={card.title} value={card.value} subtitle={card.subtitle} trend={card.trend} />
          ))}
        </section>

        <section className="mt-8 grid gap-6 lg:grid-cols-3">
          <div className="lg:col-span-2">
            <EquityCurve data={equityCurve} />
          </div>
          <div className="space-y-6">
            <SystemHealthCard metrics={summary} lastUpdated={lastUpdated} positions={positions} />
            <RiskOverviewCard metrics={summary} trades={trades} />
          </div>
        </section>

        <section className="mt-8 grid gap-6 lg:grid-cols-2">
          <PositionsTable positions={positions} />
          <TradesTable trades={trades} />
        </section>
      </main>

      <footer className="border-t border-slate-800 bg-slate-900">
        <div className="mx-auto max-w-7xl px-4 py-6 text-center text-sm text-slate-400 sm:px-6 lg:px-8">
          AGV2 Trading System v1.0.0 {' | '} Current Equity: {formatCurrency(summary.current_equity)} {' | '} Avg Trade Duration:{' '}
          {summary.avg_trade_duration_hours.toFixed(1)}h
        </div>
      </footer>
    </div>
  );
}

export default App;
