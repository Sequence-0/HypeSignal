import React from 'react';
import {
  Clock,
  Smile,
  Users,
  TrendingUp,
  Share2,
  ArrowRight,
  Flame,
  Radio,
  Sparkles,
  ShieldCheck,
  CheckCircle2,
} from 'lucide-react';
import { MetricCard } from '../components/common/MetricCard';
import { StatusBadge } from '../components/common/StatusBadge';
import { NavTab } from '../components/layout/Sidebar';
import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
} from 'recharts';
import {
  mockBurstAlerts,
  mockDemographics,
  mockKOLs,
  mockTimeseries,
} from '../services/mockData';

interface OverviewViewProps {
  setActiveTab: (tab: NavTab) => void;
}

export const OverviewView: React.FC<OverviewViewProps> = ({ setActiveTab }) => {
  return (
    <div className="space-y-6">
      {/* Top Welcome Banner */}
      <div className="bg-gradient-to-r from-indigo-900 via-indigo-800 to-slate-900 rounded-2xl p-6 text-white shadow-sm flex flex-col md:flex-row items-start md:items-center justify-between gap-4">
        <div>
          <div className="flex items-center gap-2 mb-2">
            <span className="px-2.5 py-0.5 rounded-full text-xs font-semibold bg-indigo-500/30 text-indigo-200 border border-indigo-400/30">
              Executive Pulse
            </span>
            <span className="text-xs text-indigo-200">Continuous Multi-Platform Observation</span>
          </div>
          <h2 className="text-2xl font-bold tracking-tight">
            AI-Driven Audience Intelligence & Social Flow
          </h2>
          <p className="text-sm text-indigo-100/80 mt-1 max-w-2xl">
            Real-time multi-dimensional sentiment inference, demographic profiling, trend kinematics, and follower network topology unified across X, Telegram, YouTube, Reddit, and Bluesky.
          </p>
        </div>

        <div className="flex flex-wrap gap-2.5 shrink-0">
          <button
            type="button"
            onClick={() => setActiveTab('sentiment')}
            className="px-4 py-2 bg-indigo-600 hover:bg-indigo-500 text-white rounded-xl text-xs font-semibold shadow-xs transition-colors flex items-center gap-2"
          >
            <span>Analyze Sentiment</span>
            <ArrowRight className="w-3.5 h-3.5" />
          </button>
          <button
            type="button"
            onClick={() => setActiveTab('trends')}
            className="px-4 py-2 bg-white/10 hover:bg-white/20 text-white border border-white/20 rounded-xl text-xs font-semibold transition-colors flex items-center gap-2"
          >
            <span>View Spiking Trends</span>
            <TrendingUp className="w-3.5 h-3.5" />
          </button>
        </div>
      </div>

      {/* KPI Grid (5 Core Pillars) */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-5 gap-4">
        <MetricCard
          title="Timeline Activity"
          value="12,480"
          subtitle="Tracked social posts"
          icon={Clock}
          iconColor="text-indigo-600"
          iconBg="bg-indigo-50"
          tooltip="Total posts and comments monitored with timestamped chronology."
        />
        <MetricCard
          title="Audience Mood"
          value="+0.48"
          subtitle="Dominant: Optimism (68%)"
          icon={Smile}
          iconColor="text-emerald-600"
          iconBg="bg-emerald-50"
          tooltip="Audience sentiment score (-1.0 to +1.0) with sarcasm awareness."
        />
        <MetricCard
          title="Top Demographic"
          value="25–34 yrs"
          subtitle="Tech & AI Professionals (41%)"
          icon={Users}
          iconColor="text-violet-600"
          iconBg="bg-violet-50"
          tooltip="Audience age brackets and interest clusters with privacy protection."
        />
        <MetricCard
          title="Fastest Trend"
          value="#LocalAI"
          subtitle="+5.8x volume surge"
          icon={Flame}
          iconColor="text-rose-600"
          iconBg="bg-rose-50"
          tooltip="Real-time alert detecting rapid conversation spikes."
        />
        <MetricCard
          title="Key Opinion Leaders"
          value="7,250"
          subtitle="Top: @alex_researcher (0.96)"
          icon={Share2}
          iconColor="text-amber-600"
          iconBg="bg-amber-50"
          tooltip="Influencers ranked by audience reach and network influence."
        />
      </div>

      {/* Feature Deep-Dive Cards Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Timeline Volume Quick Chart */}
        <div className="bg-white rounded-xl border border-slate-200 p-5 shadow-xs flex flex-col justify-between">
          <div className="flex items-center justify-between mb-4">
            <div>
              <h3 className="text-sm font-bold text-slate-900">Activity Volume Over Time</h3>
              <p className="text-xs text-slate-500">Hourly post count aggregated across all platforms</p>
            </div>
            <button
              type="button"
              onClick={() => setActiveTab('timeline')}
              className="text-xs text-indigo-600 hover:text-indigo-800 font-semibold flex items-center gap-1"
            >
              Full Timeline <ArrowRight className="w-3.5 h-3.5" />
            </button>
          </div>

          <div className="h-48 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={mockTimeseries}>
                <defs>
                  <linearGradient id="pulseGradient" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#6366f1" stopOpacity={0.3} />
                    <stop offset="95%" stopColor="#6366f1" stopOpacity={0} />
                  </linearGradient>
                </defs>
                <XAxis dataKey="bucket" stroke="#94a3b8" fontSize={11} tickLine={false} />
                <YAxis stroke="#94a3b8" fontSize={11} tickLine={false} axisLine={false} />
                <Tooltip
                  contentStyle={{
                    backgroundColor: '#0f172a',
                    borderRadius: '8px',
                    color: '#fff',
                    border: 'none',
                    fontSize: '12px',
                  }}
                />
                <Area
                  type="monotone"
                  dataKey="post_count"
                  stroke="#6366f1"
                  strokeWidth={2}
                  fillOpacity={1}
                  fill="url(#pulseGradient)"
                />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Spiking Trends Quick List */}
        <div className="bg-white rounded-xl border border-slate-200 p-5 shadow-xs flex flex-col justify-between">
          <div className="flex items-center justify-between mb-3">
            <div>
              <h3 className="text-sm font-bold text-slate-900">Real-Time Spiking Trends Radar</h3>
              <p className="text-xs text-slate-500">Fastest accelerating keywords and topics</p>
            </div>
            <button
              type="button"
              onClick={() => setActiveTab('trends')}
              className="text-xs text-indigo-600 hover:text-indigo-800 font-semibold flex items-center gap-1"
            >
              Explore Trends <ArrowRight className="w-3.5 h-3.5" />
            </button>
          </div>

          <div className="divide-y divide-slate-100">
            {mockBurstAlerts.slice(0, 4).map((alert) => (
              <div key={alert.term} className="py-2.5 flex items-center justify-between">
                <div>
                  <div className="flex items-center gap-2">
                    <span className="text-xs font-bold text-slate-900">{alert.term}</span>
                    <span className="text-[10px] font-semibold px-1.5 py-0.5 rounded-full bg-rose-50 text-rose-700 border border-rose-200">
                      +{alert.z_score.toFixed(1)}σ Spike
                    </span>
                  </div>
                  <span className="text-[11px] text-slate-500">
                    Current volume: {alert.current_count} posts (Baseline: {alert.baseline_mean.toFixed(0)})
                  </span>
                </div>
                <StatusBadge type="lifecycle" value="VIRAL_SURGE" />
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Cross-Pillar Quick Navigation Tiles */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <div
          onClick={() => setActiveTab('sentiment')}
          className="p-5 bg-white rounded-xl border border-slate-200 hover:border-indigo-300 hover:shadow-xs transition-all cursor-pointer group"
        >
          <div className="flex items-center justify-between mb-2">
            <div className="p-2 rounded-lg bg-emerald-50 text-emerald-600">
              <Smile className="w-5 h-5" />
            </div>
            <ArrowRight className="w-4 h-4 text-slate-400 group-hover:text-indigo-600 group-hover:translate-x-1 transition-all" />
          </div>
          <h4 className="text-sm font-bold text-slate-900">Audience Sentiment & 9 Emotions</h4>
          <p className="text-xs text-slate-500 mt-1 leading-relaxed">
            Test live texts with Cardiff RoBERTa, detect nuanced emotions (joy, anxiety, excitement), and adjust for sarcasm.
          </p>
        </div>

        <div
          onClick={() => setActiveTab('demographics')}
          className="p-5 bg-white rounded-xl border border-slate-200 hover:border-indigo-300 hover:shadow-xs transition-all cursor-pointer group"
        >
          <div className="flex items-center justify-between mb-2">
            <div className="p-2 rounded-lg bg-violet-50 text-violet-600">
              <Users className="w-5 h-5" />
            </div>
            <ArrowRight className="w-4 h-4 text-slate-400 group-hover:text-indigo-600 group-hover:translate-x-1 transition-all" />
          </div>
          <h4 className="text-sm font-bold text-slate-900">Audience Demographics & Privacy</h4>
          <p className="text-xs text-slate-500 mt-1 leading-relaxed">
            Explore age distributions, global geography, languages, and inspect influencer audiences with k-anonymity protection.
          </p>
        </div>

        <div
          onClick={() => setActiveTab('network')}
          className="p-5 bg-white rounded-xl border border-slate-200 hover:border-indigo-300 hover:shadow-xs transition-all cursor-pointer group"
        >
          <div className="flex items-center justify-between mb-2">
            <div className="p-2 rounded-lg bg-amber-50 text-amber-600">
              <Share2 className="w-5 h-5" />
            </div>
            <ArrowRight className="w-4 h-4 text-slate-400 group-hover:text-indigo-600 group-hover:translate-x-1 transition-all" />
          </div>
          <h4 className="text-sm font-bold text-slate-900">Follower Networks & Viral Cascades</h4>
          <p className="text-xs text-slate-500 mt-1 leading-relaxed">
            Uncover key opinion leaders, boundary-spanning bridge connectors, and track how information spreads across communities.
          </p>
        </div>
      </div>
    </div>
  );
};
