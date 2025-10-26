/**
 * System health card summarizing telemetry from metrics + last updated state
 */
import type { MetricsSummary, Position } from '../types';

interface SystemHealthCardProps {
  metrics: MetricsSummary;
  lastUpdated: Date | null;
  positions: Position[];
}

const formatTime = (value: Date | null) => (value ? value.toLocaleTimeString() : 'Awaiting data');

export default function SystemHealthCard({ metrics, lastUpdated, positions }: SystemHealthCardProps) {
  const equityProgress = metrics.initial_equity
    ? Math.min(150, Math.max(0, (metrics.current_equity / metrics.initial_equity) * 100))
    : 0;

  const drawdownUsage = Math.min(100, Math.max(0, Math.abs(metrics.max_drawdown)));
  const activeSymbols = positions.map((position) => position.symbol).join(', ');
  const tradingState = positions.length > 0 ? 'Trading' : 'Standing by';

  return (
    <div className="rounded-2xl border border-slate-200 bg-white p-6 shadow-sm">
      <div className="flex items-center justify-between gap-4">
        <div>
          <p className="text-xs font-semibold uppercase tracking-[0.2em] text-slate-500">System</p>
          <h3 className="text-lg font-semibold text-slate-900">Health & Telemetry</h3>
          <p className="text-sm text-slate-500">Realtime pulse of the AGV2 runner.</p>
        </div>
        <span className={`inline-flex items-center gap-2 rounded-full px-3 py-1 text-xs font-semibold ${
          positions.length ? 'bg-emerald-50 text-emerald-700' : 'bg-amber-50 text-amber-700'
        }`}>
          <span className={`h-2 w-2 rounded-full ${positions.length ? 'bg-emerald-500' : 'bg-amber-500'}`} />
          {tradingState}
        </span>
      </div>

      <dl className="mt-6 space-y-4 text-sm">
        <div className="flex items-center justify-between">
          <dt className="text-slate-500">Last sync</dt>
          <dd className="font-semibold text-slate-900">{formatTime(lastUpdated)}</dd>
        </div>
        <div className="flex items-center justify-between">
          <dt className="text-slate-500">Equity</dt>
          <dd className="font-semibold text-slate-900">
            ${metrics.current_equity.toFixed(2)}
            <span className="ml-2 text-xs font-medium text-slate-500">initial ${metrics.initial_equity.toFixed(2)}</span>
          </dd>
        </div>
        <div>
          <div className="flex items-center justify-between text-xs font-semibold text-slate-500">
            <span>Equity utilization</span>
            <span>{equityProgress.toFixed(1)}%</span>
          </div>
          <div className="mt-1 h-2 rounded-full bg-slate-100">
            <div className="h-full rounded-full bg-indigo-500" style={{ width: `${Math.min(equityProgress, 100)}%` }} aria-hidden />
          </div>
        </div>
        <div>
          <div className="flex items-center justify-between text-xs font-semibold text-slate-500">
            <span>Drawdown guard</span>
            <span>{drawdownUsage.toFixed(1)}%</span>
          </div>
          <div className="mt-1 h-2 rounded-full bg-slate-100">
            <div className="h-full rounded-full bg-rose-500" style={{ width: `${drawdownUsage}%` }} aria-hidden />
          </div>
        </div>
        <div className="flex items-center justify-between">
          <dt className="text-slate-500">Active symbols</dt>
          <dd className="font-semibold text-slate-900">{activeSymbols || 'Flat'}</dd>
        </div>
        <div className="flex items-center justify-between">
          <dt className="text-slate-500">Avg trade duration</dt>
          <dd className="font-semibold text-slate-900">{metrics.avg_trade_duration_hours.toFixed(1)}h</dd>
        </div>
      </dl>
    </div>
  );
}
