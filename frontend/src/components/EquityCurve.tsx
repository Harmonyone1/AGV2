/**
 * Equity curve chart component with lightweight range filtering
 */
import { useMemo, useState } from 'react';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend,
} from 'recharts';
import type { EquityCurvePoint } from '../types';

type RangeKey = '1W' | '1M' | '3M' | 'ALL';
const ranges: Record<RangeKey, number | null> = {
  '1W': 7,
  '1M': 30,
  '3M': 90,
  ALL: null,
};

const currencyFormatter = (value: number) =>
  new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD', maximumFractionDigits: 0 }).format(value ?? 0);

interface EquityCurveProps {
  data: EquityCurvePoint[];
}

export default function EquityCurve({ data }: EquityCurveProps) {
  const [activeRange, setActiveRange] = useState<RangeKey>('1M');

  const chartData = useMemo(
    () =>
      data.map((point) => {
        const date = new Date(point.timestamp);
        return {
          label: date.toLocaleDateString(),
          timestamp: date.getTime(),
          equity: point.equity,
          pnl: point.pnl,
        };
      }),
    [data],
  );

  const filteredData = useMemo(() => {
    const windowDays = ranges[activeRange];
    if (!windowDays) return chartData;
    const cutoff = Date.now() - windowDays * 24 * 60 * 60 * 1000;
    return chartData.filter((point) => point.timestamp >= cutoff);
  }, [chartData, activeRange]);

  if (filteredData.length === 0) {
    return (
      <div className="rounded-2xl border border-dashed border-slate-200 bg-white/60 p-6 text-center shadow-sm">
        <p className="text-base font-semibold text-slate-600">No equity data yet</p>
        <p className="mt-2 text-sm text-slate-500">Run a backtest or connect to the live engine to populate this chart.</p>
      </div>
    );
  }

  return (
    <div className="rounded-2xl border border-slate-200 bg-white p-6 shadow-sm">
      <div className="flex flex-wrap items-center justify-between gap-4">
        <div>
          <p className="text-xs font-semibold uppercase tracking-[0.2em] text-slate-500">Performance</p>
          <h3 className="text-xl font-semibold text-slate-900">Equity Curve</h3>
          <p className="text-sm text-slate-500">Smoothed account value with intraday P&L.</p>
        </div>
        <div className="flex items-center gap-2">
          {(Object.keys(ranges) as RangeKey[]).map((range) => (
            <button
              key={range}
              type="button"
              onClick={() => setActiveRange(range)}
              className={`rounded-full px-3 py-1 text-xs font-semibold transition ${
                activeRange === range ? 'bg-indigo-600 text-white shadow' : 'bg-slate-100 text-slate-600'
              }`}
            >
              {range}
            </button>
          ))}
        </div>
      </div>
      <div className="mt-6 h-[320px] w-full">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={filteredData} margin={{ top: 5, right: 20, bottom: 0, left: 0 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
            <XAxis dataKey="label" tick={{ fill: '#475569', fontSize: 12 }} axisLine={false} tickLine={false} minTickGap={24} />
            <YAxis tickFormatter={(value) => currencyFormatter(value)} tick={{ fill: '#475569', fontSize: 12 }} axisLine={false} tickLine={false} />
            <Tooltip
              content={({ active, payload, label }) => {
                if (!active || !payload?.length) return null;
                const [equityPoint, pnlPoint] = payload;
                return (
                  <div className="rounded-lg border border-slate-200 bg-white px-4 py-3 text-sm shadow-lg">
                    <p className="font-semibold text-slate-900">{label}</p>
                    <p className="mt-1 text-slate-600">Equity: {currencyFormatter(equityPoint.value as number)}</p>
                    {pnlPoint && <p className="text-slate-500">P&L: {currencyFormatter(pnlPoint.value as number)}</p>}
                  </div>
                );
              }}
            />
            <Legend verticalAlign="top" align="right" iconType="circle" wrapperStyle={{ paddingBottom: 10 }} />
            <Line type="monotone" dataKey="equity" stroke="#4f46e5" strokeWidth={2.5} dot={false} name="Equity" />
            <Line type="basis" dataKey="pnl" stroke="#10b981" strokeWidth={2} dot={false} strokeDasharray="6 6" name="Daily P&L" />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
