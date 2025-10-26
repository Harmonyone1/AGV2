/**
 * Recent trades table component with additional visual cues
 */
import type { Trade } from '../types';

interface TradesTableProps {
  trades: Trade[];
}

const formatCurrency = (value: number) =>
  new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD', maximumFractionDigits: 2 }).format(value ?? 0);

const pctFormatter = (value: number) => `${value.toFixed(2)}%`;

export default function TradesTable({ trades }: TradesTableProps) {
  if (trades.length === 0) {
    return (
      <div className="rounded-2xl border border-slate-200 bg-white p-6 shadow-sm">
        <div className="flex items-center justify-between">
          <div>
            <p className="text-xs font-semibold uppercase tracking-[0.2em] text-slate-500">Execution</p>
            <h3 className="text-lg font-semibold text-slate-900">Recent Trades</h3>
          </div>
          <span className="rounded-full bg-slate-100 px-3 py-1 text-xs font-semibold text-slate-600">Awaiting fills</span>
        </div>
        <div className="mt-6 rounded-2xl border border-dashed border-slate-200 bg-slate-50/70 p-6 text-center">
          <p className="text-base font-semibold text-slate-700">No trades logged</p>
          <p className="mt-2 text-sm text-slate-500">Monitor the live engine or rerun a simulation to populate fills.</p>
        </div>
      </div>
    );
  }

  return (
    <div className="rounded-2xl border border-slate-200 bg-white p-6 shadow-sm">
      <div className="flex items-center justify-between">
        <div>
          <p className="text-xs font-semibold uppercase tracking-[0.2em] text-slate-500">Execution</p>
          <h3 className="text-lg font-semibold text-slate-900">Recent Trades</h3>
          <p className="text-sm text-slate-500">Last {trades.length} fills</p>
        </div>
        <span className="rounded-full bg-indigo-50 px-3 py-1 text-xs font-semibold text-indigo-700">Live feed</span>
      </div>
      <div className="mt-6 overflow-x-auto rounded-xl border border-slate-100">
        <table className="min-w-full divide-y divide-slate-100 text-left text-sm">
          <thead className="bg-slate-50 text-xs font-semibold uppercase tracking-wide text-slate-500">
            <tr>
              <th className="px-6 py-3">Symbol</th>
              <th className="px-6 py-3">Entry</th>
              <th className="px-6 py-3">Exit</th>
              <th className="px-6 py-3">P&L</th>
              <th className="px-6 py-3">Duration</th>
              <th className="px-6 py-3">Exit reason</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-100 bg-white">
            {trades.map((trade, index) => (
              <tr key={`${trade.symbol}-${index}`} className="hover:bg-slate-50/80">
                <td className="px-6 py-4 text-sm font-semibold text-slate-900">{trade.symbol}</td>
                <td className="px-6 py-4 text-sm text-slate-600">{formatCurrency(trade.entry_price)}</td>
                <td className="px-6 py-4 text-sm text-slate-600">{formatCurrency(trade.exit_price)}</td>
                <td className={`px-6 py-4 text-sm font-semibold ${trade.pnl >= 0 ? 'text-emerald-600' : 'text-rose-600'}`}>
                  {formatCurrency(trade.pnl)} ({pctFormatter(trade.pnl_pct)})
                </td>
                <td className="px-6 py-4 text-sm text-slate-600">{trade.duration_hours.toFixed(1)}h</td>
                <td className="px-6 py-4 text-sm text-slate-600">{trade.exit_reason}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
