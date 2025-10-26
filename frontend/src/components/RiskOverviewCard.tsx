/**
 * Risk overview card summarizing PnL + risk thresholds
 */
import type { MetricsSummary, Trade } from '../types';

interface RiskOverviewCardProps {
  metrics: MetricsSummary;
  trades: Trade[];
}

const formatCurrency = (value: number) =>
  new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD', maximumFractionDigits: 2 }).format(value ?? 0);

export default function RiskOverviewCard({ metrics, trades }: RiskOverviewCardProps) {
  const todayPositive = metrics.today_pnl >= 0;
  const winRatePct = (metrics.win_rate * 100).toFixed(1);
  const latestTrade = trades[0];

  return (
    <div className="rounded-2xl border border-slate-200 bg-white p-6 shadow-sm">
      <div className="flex items-center justify-between">
        <div>
          <p className="text-xs font-semibold uppercase tracking-[0.2em] text-slate-500">Risk</p>
          <h3 className="text-lg font-semibold text-slate-900">Risk Envelope</h3>
          <p className="text-sm text-slate-500">Limits vs. current performance</p>
        </div>
        <span className={`rounded-full px-3 py-1 text-xs font-semibold ${
          todayPositive ? 'bg-emerald-50 text-emerald-700' : 'bg-rose-50 text-rose-700'
        }`}>
          {todayPositive ? 'Within guardrails' : 'Monitor losses'}
        </span>
      </div>

      <div className="mt-6 space-y-4 text-sm">
        <div className="flex items-center justify-between">
          <p className="text-slate-500">Today's P&L</p>
          <div className={`font-semibold ${todayPositive ? 'text-emerald-600' : 'text-rose-600'}`}>
            {formatCurrency(metrics.today_pnl)} ({metrics.today_pnl_pct.toFixed(2)}%)
          </div>
        </div>
        <div className="flex items-center justify-between">
          <p className="text-slate-500">Total P&L</p>
          <div className={`font-semibold ${metrics.total_pnl >= 0 ? 'text-emerald-600' : 'text-rose-600'}`}>
            {formatCurrency(metrics.total_pnl)} ({metrics.total_pnl_pct.toFixed(2)}%)
          </div>
        </div>
        <div className="flex items-center justify-between">
          <p className="text-slate-500">Max drawdown</p>
          <div className="font-semibold text-slate-900">{metrics.max_drawdown.toFixed(2)}%</div>
        </div>
        <div className="flex items-center justify-between">
          <p className="text-slate-500">Win rate</p>
          <div className="font-semibold text-slate-900">{winRatePct}%</div>
        </div>
        <div className="rounded-xl border border-slate-100 bg-slate-50/70 p-4">
          <p className="text-xs font-semibold uppercase tracking-[0.2em] text-slate-500">Latest trade</p>
          {latestTrade ? (
            <div className="mt-2 text-sm">
              <p className="font-semibold text-slate-900">{latestTrade.symbol}</p>
              <p className="text-slate-500">
                {latestTrade.position_size.toFixed(2)} @ {formatCurrency(latestTrade.entry_price)} {' -> '} {formatCurrency(latestTrade.exit_price)}
              </p>
              <p className={`mt-1 font-semibold ${latestTrade.pnl >= 0 ? 'text-emerald-600' : 'text-rose-600'}`}>
                {formatCurrency(latestTrade.pnl)} ({latestTrade.pnl_pct.toFixed(2)}%) {' | '} {latestTrade.exit_reason}
              </p>
            </div>
          ) : (
            <p className="mt-2 text-sm text-slate-500">No fills yet today.</p>
          )}
        </div>
      </div>
    </div>
  );
}
