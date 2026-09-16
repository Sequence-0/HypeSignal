import React from 'react';
import { Flame, TrendingUp, Zap, Clock, ShieldCheck, CheckCircle2, AlertCircle } from 'lucide-react';
import { TrendLifecycleState } from '../../types';

interface StatusBadgeProps {
  type: 'sentiment' | 'lifecycle' | 'platform' | 'privacy' | 'status';
  value: string;
  className?: string;
}

export const StatusBadge: React.FC<StatusBadgeProps> = ({ type, value, className = '' }) => {
  const v = value.toLowerCase();

  if (type === 'sentiment') {
    if (v.includes('pos')) {
      return (
        <span className={`inline-flex items-center gap-1 px-2.5 py-0.5 rounded-full text-xs font-medium bg-emerald-50 text-emerald-700 border border-emerald-200 ${className}`}>
          <span className="w-1.5 h-1.5 rounded-full bg-emerald-500" />
          Positive
        </span>
      );
    }
    if (v.includes('neg')) {
      return (
        <span className={`inline-flex items-center gap-1 px-2.5 py-0.5 rounded-full text-xs font-medium bg-rose-50 text-rose-700 border border-rose-200 ${className}`}>
          <span className="w-1.5 h-1.5 rounded-full bg-rose-500" />
          Negative
        </span>
      );
    }
    return (
      <span className={`inline-flex items-center gap-1 px-2.5 py-0.5 rounded-full text-xs font-medium bg-slate-100 text-slate-700 border border-slate-200 ${className}`}>
        <span className="w-1.5 h-1.5 rounded-full bg-slate-400" />
        Neutral
      </span>
    );
  }

  if (type === 'lifecycle') {
    const state = value as TrendLifecycleState;
    switch (state) {
      case 'VIRAL_SURGE':
        return (
          <span className={`inline-flex items-center gap-1 px-2.5 py-1 rounded-full text-xs font-semibold bg-rose-50 text-rose-700 border border-rose-200 shadow-xs ${className}`}>
            <Flame className="w-3.5 h-3.5 text-rose-500 animate-pulse" />
            Viral Surge
          </span>
        );
      case 'EMERGING':
        return (
          <span className={`inline-flex items-center gap-1 px-2.5 py-1 rounded-full text-xs font-semibold bg-indigo-50 text-indigo-700 border border-indigo-200 ${className}`}>
            <TrendingUp className="w-3.5 h-3.5 text-indigo-500" />
            Emerging
          </span>
        );
      case 'PEAKING':
        return (
          <span className={`inline-flex items-center gap-1 px-2.5 py-1 rounded-full text-xs font-semibold bg-amber-50 text-amber-700 border border-amber-200 ${className}`}>
            <Zap className="w-3.5 h-3.5 text-amber-500" />
            At Peak
          </span>
        );
      case 'DECELERATING':
        return (
          <span className={`inline-flex items-center gap-1 px-2.5 py-1 rounded-full text-xs font-medium bg-slate-100 text-slate-600 border border-slate-200 ${className}`}>
            <Clock className="w-3.5 h-3.5 text-slate-400" />
            Cooling Down
          </span>
        );
      default:
        return (
          <span className={`inline-flex items-center gap-1 px-2.5 py-1 rounded-full text-xs font-medium bg-slate-50 text-slate-600 border border-slate-200 ${className}`}>
            Stable
          </span>
        );
    }
  }

  if (type === 'platform') {
    const colors: Record<string, { bg: string; text: string; border: string }> = {
      twitter: { bg: 'bg-sky-50', text: 'text-sky-700', border: 'border-sky-200' },
      telegram: { bg: 'bg-blue-50', text: 'text-blue-700', border: 'border-blue-200' },
      youtube: { bg: 'bg-red-50', text: 'text-red-700', border: 'border-red-200' },
      reddit: { bg: 'bg-orange-50', text: 'text-orange-700', border: 'border-orange-200' },
      bluesky: { bg: 'bg-cyan-50', text: 'text-cyan-700', border: 'border-cyan-200' },
    };
    const c = colors[v] || { bg: 'bg-slate-50', text: 'text-slate-700', border: 'border-slate-200' };
    const label = v === 'twitter' ? 'X (Twitter)' : v.charAt(0).toUpperCase() + v.slice(1);
    return (
      <span className={`inline-flex items-center px-2 py-0.5 rounded text-xs font-medium border ${c.bg} ${c.text} ${c.border} ${className}`}>
        {label}
      </span>
    );
  }

  if (type === 'privacy') {
    return (
      <span className={`inline-flex items-center gap-1 px-2.5 py-0.5 rounded-full text-xs font-semibold bg-emerald-50 text-emerald-800 border border-emerald-200 ${className}`}>
        <ShieldCheck className="w-3.5 h-3.5 text-emerald-600" />
        Privacy Shield Active (k ≥ 5)
      </span>
    );
  }

  if (type === 'status') {
    const isOnline = v.includes('online') || v.includes('true') || v.includes('ok');
    return (
      <span className={`inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-xs font-medium ${isOnline ? 'bg-emerald-50 text-emerald-700 border border-emerald-200' : 'bg-rose-50 text-rose-700 border border-rose-200'} ${className}`}>
        {isOnline ? <CheckCircle2 className="w-3 h-3 text-emerald-500" /> : <AlertCircle className="w-3 h-3 text-rose-500" />}
        {value}
      </span>
    );
  }

  return (
    <span className={`inline-flex items-center px-2 py-0.5 rounded text-xs font-medium bg-slate-100 text-slate-800 border border-slate-200 ${className}`}>
      {value}
    </span>
  );
};
