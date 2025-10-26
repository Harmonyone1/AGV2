/**
 * Metric card component for displaying key metrics in a richer format
 */
interface MetricCardProps {
  title: string;
  value: string | number;
  subtitle?: string;
  trend?: 'up' | 'down' | 'neutral';
}

const trendTokens = {
  up: {
    label: 'Bullish',
    badge: 'bg-emerald-50 text-emerald-700',
    accent: 'text-emerald-600',
    icon: (
      <svg viewBox="0 0 20 20" fill="currentColor" className="h-4 w-4">
        <path d="M3 12.5 9 6.5 13 10.5 17 6.5" stroke="currentColor" strokeWidth="2" fill="none" strokeLinecap="round" strokeLinejoin="round" />
        <path d="M17 6.5v6" stroke="currentColor" strokeWidth="2" strokeLinecap="round" />
      </svg>
    ),
  },
  down: {
    label: 'Drawdown',
    badge: 'bg-rose-50 text-rose-700',
    accent: 'text-rose-600',
    icon: (
      <svg viewBox="0 0 20 20" fill="currentColor" className="h-4 w-4">
        <path d="M3 7.5 9 13.5 13 9.5 17 13.5" stroke="currentColor" strokeWidth="2" fill="none" strokeLinecap="round" strokeLinejoin="round" />
        <path d="M17 13.5v-6" stroke="currentColor" strokeWidth="2" strokeLinecap="round" />
      </svg>
    ),
  },
  neutral: {
    label: 'Stable',
    badge: 'bg-slate-100 text-slate-700',
    accent: 'text-slate-500',
    icon: (
      <svg viewBox="0 0 20 20" fill="currentColor" className="h-4 w-4">
        <path d="M4 10h12" stroke="currentColor" strokeWidth="2" strokeLinecap="round" />
      </svg>
    ),
  },
};

export default function MetricCard({ title, value, subtitle, trend = 'neutral' }: MetricCardProps) {
  const token = trendTokens[trend];

  return (
    <div className="relative overflow-hidden rounded-2xl border border-slate-100 bg-white p-5 shadow-sm">
      <div className="absolute -right-6 -top-6 h-20 w-20 rounded-full bg-indigo-50" aria-hidden />
      <div className="flex items-start justify-between gap-2">
        <div>
          <p className="text-xs font-semibold uppercase tracking-[0.2em] text-slate-500">{title}</p>
          <p className="mt-3 text-3xl font-semibold text-slate-900">{value}</p>
        </div>
        <span className={`inline-flex items-center gap-1 rounded-full px-3 py-1 text-xs font-semibold ${token.badge}`}>
          {token.icon}
          {token.label}
        </span>
      </div>
      {subtitle && <p className="mt-4 text-sm font-medium text-slate-600">{subtitle}</p>}
      <div className={`mt-6 h-1 rounded-full bg-slate-100`}>
        <div className={`h-full rounded-full ${token.accent}`} style={{ width: '65%' }} aria-hidden />
      </div>
    </div>
  );
}
