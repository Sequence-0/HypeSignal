import React, { useState } from 'react';
import {
  TrendingUp,
  Flame,
  Zap,
  Clock,
  Compass,
  ArrowUpRight,
  ArrowDownRight,
  Search,
  Sparkles,
  Layers,
  Activity,
} from 'lucide-react';
import { StatusBadge } from '../components/common/StatusBadge';
import {
  mockBurstAlerts,
  mockDynamicTopics,
  mockNarrativeDrift,
  mockTrendForecasts,
} from '../services/mockData';

export const TrendsView: React.FC = () => {
  const [selectedTopic, setSelectedTopic] = useState(mockTrendForecasts[0]);
  const [forecastInput, setForecastInput] = useState('#LocalAI');

  return (
    <div className="space-y-6">
      {/* Title */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div>
          <div className="flex items-center gap-2">
            <span className="px-2.5 py-0.5 rounded-full text-xs font-semibold bg-rose-50 text-rose-700 border border-rose-200">
              Trends & Momentum
            </span>
            <h2 className="text-xl font-bold text-slate-900">
              Trending Topics & Viral Momentum
            </h2>
          </div>
          <p className="text-xs text-slate-500 mt-1">
            Real-time conversation spikes, growth speed & momentum forecasting, and evolving discussion topics.
          </p>
        </div>

        <div className="flex items-center gap-2">
          <span className="px-3 py-1 bg-white border border-slate-200 rounded-lg text-xs font-semibold text-slate-700 shadow-2xs">
            Live Trend Detection Active
          </span>
        </div>
      </div>

      {/* Kinematics Forecaster / Virality Gauge Card */}
      <div className="bg-gradient-to-r from-slate-900 via-indigo-950 to-slate-900 rounded-2xl p-6 text-white shadow-sm">
        <div className="flex flex-col lg:flex-row items-start lg:items-center justify-between gap-6">
          <div className="space-y-2">
            <div className="flex items-center gap-2">
              <span className="px-2.5 py-0.5 rounded-full text-[11px] font-bold bg-rose-500/20 text-rose-300 border border-rose-500/30">
                Momentum Forecaster
              </span>
              <span className="text-xs text-slate-400">Speed & Acceleration Metrics</span>
            </div>
            <h3 className="text-2xl font-bold tracking-tight">
              Tracking &quot;{selectedTopic.term}&quot;
            </h3>
            <p className="text-xs text-slate-300 max-w-xl leading-relaxed">
              Real-time velocity of {selectedTopic.velocity} posts/min with acceleration of +{selectedTopic.acceleration} posts/min². High author diversity ({Math.round(selectedTopic.author_diversity_ratio * 100)}%) confirms broad community participation.
            </p>
          </div>

          <div className="flex flex-wrap items-center gap-4 w-full lg:w-auto">
            {/* Virality Potential Meter */}
            <div className="bg-white/10 border border-white/15 rounded-xl p-4 flex-1 lg:flex-none text-center min-w-[140px]">
              <div className="text-[11px] font-semibold text-slate-300 uppercase tracking-wider">
                Virality Chance
              </div>
              <div className="text-3xl font-extrabold text-white mt-1">
                {(selectedTopic.virality_potential_score * 100).toFixed(0)}%
              </div>
              <div className="text-[10px] text-emerald-400 font-semibold mt-0.5">High Potential</div>
            </div>

            {/* Growth Speed */}
            <div className="bg-white/10 border border-white/15 rounded-xl p-4 flex-1 lg:flex-none text-center min-w-[140px]">
              <div className="text-[11px] font-semibold text-slate-300 uppercase tracking-wider">
                Growth Speed
              </div>
              <div className="text-2xl font-bold text-white mt-1">
                +{selectedTopic.velocity}
              </div>
              <div className="text-[10px] text-slate-400 mt-0.5">posts per minute</div>
            </div>

            {/* Growth Momentum (Acceleration) */}
            <div className="bg-white/10 border border-white/15 rounded-xl p-4 flex-1 lg:flex-none text-center min-w-[140px]">
              <div className="text-[11px] font-semibold text-slate-300 uppercase tracking-wider">
                Momentum
              </div>
              <div className="text-2xl font-bold text-white mt-1">
                +{selectedTopic.acceleration}
              </div>
              <div className="text-[10px] text-slate-400 mt-0.5">acceleration rate</div>
            </div>
          </div>
        </div>

        {/* Quick Topic Switcher */}
        <div className="mt-6 pt-4 border-t border-white/10 flex flex-wrap items-center gap-2">
          <span className="text-xs text-slate-400 font-medium">Select Active Term:</span>
          {mockTrendForecasts.map((t) => (
            <button
              key={t.term}
              type="button"
              onClick={() => setSelectedTopic(t)}
              className={`px-3 py-1 rounded-lg text-xs font-semibold transition-all ${
                selectedTopic.term === t.term
                  ? 'bg-indigo-600 text-white shadow-xs'
                  : 'bg-white/10 hover:bg-white/20 text-slate-300'
              }`}
            >
              {t.term}
            </button>
          ))}
        </div>
      </div>

      {/* Spiking Trends Radar (Tier-1 Burstiness Alerts) */}
      <div className="bg-white rounded-xl border border-slate-200 overflow-hidden shadow-xs">
        <div className="p-5 bg-slate-50 border-b border-slate-200 flex items-center justify-between">
          <div>
            <div className="flex items-center gap-2">
              <Flame className="w-5 h-5 text-rose-500" />
              <h3 className="text-sm font-bold text-slate-900">
                Spiking Trends Radar (Tier-1 Statistical Bursts)
              </h3>
            </div>
            <p className="text-xs text-slate-500 mt-0.5">
              Instant alerts when chatter volume exceeds rolling historical moving baseline (Z-score &gt; 2.5σ)
            </p>
          </div>
          <span className="text-xs font-semibold px-2.5 py-1 bg-rose-50 text-rose-700 border border-rose-200 rounded-full">
            {mockBurstAlerts.length} Active Spikes
          </span>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs">
            <thead className="bg-slate-50/70 border-b border-slate-200 text-slate-500 font-semibold uppercase tracking-wider text-[11px]">
              <tr>
                <th className="py-3 px-4">Trending Keyword / Topic</th>
                <th className="py-3 px-4">Current Volume</th>
                <th className="py-3 px-4">Baseline Normal</th>
                <th className="py-3 px-4">Spike Intensity (Z-Score)</th>
                <th className="py-3 px-4">Lifecycle State</th>
                <th className="py-3 px-4 text-right">Action</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100">
              {mockBurstAlerts.map((alert) => (
                <tr key={alert.term} className="hover:bg-slate-50/80 transition-colors">
                  <td className="py-3 px-4 font-bold text-slate-900 flex items-center gap-2">
                    <span className="w-2 h-2 rounded-full bg-rose-500" />
                    {alert.term}
                  </td>
                  <td className="py-3 px-4 font-semibold text-slate-800">
                    {alert.current_count} posts
                  </td>
                  <td className="py-3 px-4 text-slate-500">
                    {alert.baseline_mean.toFixed(0)} ± {alert.baseline_std.toFixed(0)}
                  </td>
                  <td className="py-3 px-4">
                    <span className="inline-flex items-center gap-1 font-bold text-rose-600 bg-rose-50 px-2 py-0.5 rounded border border-rose-200">
                      <ArrowUpRight className="w-3.5 h-3.5" />
                      +{alert.z_score.toFixed(2)}σ
                    </span>
                  </td>
                  <td className="py-3 px-4">
                    <StatusBadge type="lifecycle" value="VIRAL_SURGE" />
                  </td>
                  <td className="py-3 px-4 text-right">
                    <button
                      type="button"
                      onClick={() => {
                        const match = mockTrendForecasts.find((f) => f.term === alert.term);
                        if (match) setSelectedTopic(match);
                      }}
                      className="text-xs font-semibold text-indigo-600 hover:text-indigo-800"
                    >
                      Forecast Kinematics
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Dynamic Evolving Topics (Tier-2 BERTopic) */}
      <div className="bg-white rounded-xl border border-slate-200 p-5 shadow-xs">
        <div className="flex items-center justify-between mb-4">
          <div>
            <div className="flex items-center gap-2">
              <Layers className="w-5 h-5 text-indigo-600" />
              <h3 className="text-sm font-bold text-slate-900">
                Dynamic Evolving Conversation Themes (Tier-2 Modeler)
              </h3>
            </div>
            <p className="text-xs text-slate-500 mt-0.5">
              Unsupervised clustering via BERTopic and c-TF-IDF representations across historical sliding windows
            </p>
          </div>
          <span className="text-xs font-semibold px-2.5 py-1 bg-indigo-50 text-indigo-700 border border-indigo-200 rounded-full">
            {mockDynamicTopics.length} Dynamic Clusters
          </span>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          {mockDynamicTopics.map((topic) => (
            <div
              key={topic.topic_id}
              className="p-4 bg-slate-50 rounded-xl border border-slate-200 flex flex-col justify-between"
            >
              <div>
                <div className="flex items-center justify-between mb-2">
                  <span className="text-xs font-bold text-indigo-700">Topic #{topic.topic_id}</span>
                  <span className="text-[11px] font-medium text-slate-500">
                    {topic.doc_count.toLocaleString()} posts
                  </span>
                </div>
                <h4 className="text-xs font-bold text-slate-900 leading-snug">
                  {topic.name}
                </h4>

                <div className="flex flex-wrap gap-1.5 mt-3">
                  {topic.top_words.map((w) => (
                    <span
                      key={w.word}
                      className="px-2 py-0.5 rounded text-[10px] font-medium bg-white border border-slate-200 text-slate-700"
                    >
                      {w.word}
                    </span>
                  ))}
                </div>
              </div>

              <div className="mt-4 pt-3 border-t border-slate-200 text-[11px] text-slate-500 flex justify-between">
                <span>Cluster Weight:</span>
                <span className="font-semibold text-slate-800">
                  {topic.top_words[0]?.weight.toFixed(2)}
                </span>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Narrative Concept Drift Tracker */}
      <div className="bg-white rounded-xl border border-slate-200 p-5 shadow-xs">
        <div className="flex items-center gap-2 mb-3">
          <Compass className="w-5 h-5 text-indigo-600" />
          <h3 className="text-sm font-bold text-slate-900">
            Narrative Shift & Sentiment Inversion Tracker
          </h3>
        </div>
        <p className="text-xs text-slate-500 mb-4">
          Tracks semantic centroid drift and sentiment swings between early baseline and current observation windows.
        </p>

        <div className="p-4 bg-amber-50/50 rounded-xl border border-amber-200 flex flex-col md:flex-row items-start md:items-center justify-between gap-4">
          <div className="space-y-1">
            <div className="flex items-center gap-2">
              <span className="text-xs font-bold text-amber-900">
                Drift Detected on &quot;{mockNarrativeDrift.term}&quot;
              </span>
              <span className="px-2 py-0.5 rounded-full text-[10px] font-bold bg-rose-100 text-rose-800 border border-rose-200">
                Sentiment Inverted
              </span>
            </div>
            <p className="text-xs text-amber-800 leading-relaxed max-w-2xl">
              Public tone shifted from positive optimism (+0.45) in the initial baseline window to critical anxiety (-0.38) in recent discussions, accompanied by a 64% semantic centroid drift.
            </p>
          </div>

          <div className="flex items-center gap-4 shrink-0 text-xs">
            <div className="px-3 py-2 bg-white rounded-lg border border-amber-200 text-center">
              <span className="text-slate-400 block text-[10px]">Baseline Tone</span>
              <span className="font-bold text-emerald-600">+{mockNarrativeDrift.baseline_sentiment.toFixed(2)}</span>
            </div>
            <div className="text-slate-400 font-bold">→</div>
            <div className="px-3 py-2 bg-white rounded-lg border border-amber-200 text-center">
              <span className="text-slate-400 block text-[10px]">Current Tone</span>
              <span className="font-bold text-rose-600">{mockNarrativeDrift.current_sentiment.toFixed(2)}</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
