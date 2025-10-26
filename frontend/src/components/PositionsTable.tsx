/**
 * Active positions table component with richer styling
 */
import type { Position } from '../types';

interface PositionsTableProps {
  positions: Position[];
}

const formatCurrency = (value: number) =>
  new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD', maximumFractionDigits: 2 }).format(value ?? 0);

const pctFormatter = (value: number) => `${value.toFixed(2)}%`;

export default function PositionsTable({ positions }: PositionsTableProps) {
  if (positions.length === 0) {
    return (
      <div className="rounded-2xl border border-slate-200 bg-white p-6 shadow-sm">
        <div className="flex items-center justify-between">
          <div>
            <p className="text-xs font-semibold uppercase tracking-[0.2em] text-slate-500">Positions</p>
            <h3 className="text-lg font-semibold text-slate-900">Active Positions</h3>
          </div>
          <span className="rounded-full bg-slate-100 px-3 py-1 text-xs font-semibold text-slate-600">Flat</span>
        </div>
        <div className="mt-6 rounded-2xl border border-dashed border-slate-200 bg-slate-50/70 p-6 text-center">
          <p className="text-base font-semibold text-slate-700">No active positions</p>
          <p className="mt-2 text-sm text-slate-500">The strategy is waiting for the next high-confidence signal.</p>
        </div>
      </div>
    );
  }

  return (
    <div className="rounded-2xl border border-slate-200 bg-white p-6 shadow-sm">
      <div className="flex items-center justify-between">
        <div>
          <p className="text-xs font-semibold uppercase tracking-[0.2em] text-slate-500">Positions</p>
          <h3 className="text-lg font-semibold text-slate-900">Active Positions</h3>
          <p className="text-sm text-slate-500">{positions.length} open {positions.length === 1 ? 'position' : 'positions'}</p>
        </div>
        <span className="rounded-full bg-emerald-50 px-3 py-1 text-xs font-semibold text-emerald-700">Live</span>
      </div>
      <div className="mt-6 overflow-x-auto rounded-xl border border-slate-100">
        <table className="min-w-full divide-y divide-slate-100 text-left text-sm">
          <thead className="bg-slate-50 text-xs font-semibold uppercase tracking-wide text-slate-500">
            <tr>
              <th className="px-6 py-3">Symbol</th>
              <th className="px-6 py-3">Entry</th>
              <th className="px-6 py-3">Current</th>
              <th className="px-6 py-3">Size</th>
              <th className="px-6 py-3">Unrealized</th>
              <th className="px-6 py-3">Duration</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-100 bg-white">
            {positions.map((position, index) => {
              const isLong = position.position_size >= 0;
              return (
                <tr key={`${position.symbol}-${index}`} className="hover:bg-slate-50/80">
                  <td className="px-6 py-4 text-sm font-semibold text-slate-900">
                    <div className="flex items-center gap-2">
                      <span className={`inline-flex items-center rounded-full px-2.5 py-0.5 text-xs font-semibold ${
                        isLong ? 'bg-emerald-50 text-emerald-700' : 'bg-rose-50 text-rose-700'
                      }`}>
                        {isLong ? 'Long' : 'Short'}
                      </span>
                      {position.symbol}
                    </div>
                  </td>
                  <td className="px-6 py-4 text-sm text-slate-600">{formatCurrency(position.entry_price)}</td>
                  <td className="px-6 py-4 text-sm text-slate-600">{formatCurrency(position.current_price)}</td>
                  <td className="px-6 py-4 text-sm text-slate-600">{position.position_size.toFixed(2)}</td>
                  <td className={`px-6 py-4 text-sm font-semibold ${
                    position.unrealized_pnl >= 0 ? 'text-emerald-600' : 'text-rose-600'
                  }`}>
                    {formatCurrency(position.unrealized_pnl)} ({pctFormatter(position.unrealized_pnl_pct)})
                  </td>
                  <td className="px-6 py-4 text-sm text-slate-600">{position.duration_hours.toFixed(1)}h</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}
