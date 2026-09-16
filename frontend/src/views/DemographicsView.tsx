import React, { useState } from 'react';
import {
  Users,
  Globe,
  MapPin,
  Languages,
  Briefcase,
  ShieldCheck,
  Search,
  CheckCircle2,
  Calendar,
} from 'lucide-react';
import { StatusBadge } from '../components/common/StatusBadge';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  PieChart,
  Pie,
  Cell,
} from 'recharts';
import {
  mockDemographics,
  mockInfluencerAudience,
  mockProfiledUsers,
} from '../services/mockData';

const COLORS = ['#6366f1', '#10b981', '#f59e0b', '#ec4899', '#8b5cf6', '#06b6d4'];

export const DemographicsView: React.FC = () => {
  const [selectedInfluencer, setSelectedInfluencer] = useState('alex_researcher');
  const [minGroupSize, setMinGroupSize] = useState(5);

  const ageData = Object.entries(mockDemographics.age_distribution).map(([bracket, count]) => ({
    bracket,
    count,
    percentage: Math.round((count / mockDemographics.total_users_profiled) * 100),
  }));

  const personaData = Object.entries(mockDemographics.persona_distribution).map(
    ([persona, count]) => ({
      persona,
      count,
      percentage: Math.round((count / mockDemographics.total_users_profiled) * 100),
    })
  );

  return (
    <div className="space-y-6">
      {/* Title */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div>
          <div className="flex items-center gap-2">
            <span className="px-2.5 py-0.5 rounded-full text-xs font-semibold bg-violet-50 text-violet-700 border border-violet-200">
              Audience Demographics
            </span>
            <h2 className="text-xl font-bold text-slate-900">
              Audience Demographics & Follower Insights
            </h2>
          </div>
          <p className="text-xs text-slate-500 mt-1">
            Aggregate anonymized audience breakdown: age groups, geographic locations, languages, and professional interest clusters.
          </p>
        </div>

        <StatusBadge type="privacy" value="Active" />
      </div>

      {/* Aggregate KPI Summary */}
      <div className="grid grid-cols-1 sm:grid-cols-4 gap-4">
        <div className="bg-white p-4 rounded-xl border border-slate-200 shadow-xs">
          <span className="text-xs text-slate-500 font-medium">Audited User Population</span>
          <div className="text-2xl font-bold text-slate-900 mt-1">
            {mockDemographics.total_users_profiled.toLocaleString()}
          </div>
          <span className="text-[11px] text-slate-400">Public profile indicators</span>
        </div>

        <div className="bg-white p-4 rounded-xl border border-slate-200 shadow-xs">
          <span className="text-xs text-slate-500 font-medium">Predominant Age Group</span>
          <div className="text-2xl font-bold text-slate-900 mt-1">25–34 yrs</div>
          <span className="text-[11px] text-indigo-600 font-medium">43.2% of total audience</span>
        </div>

        <div className="bg-white p-4 rounded-xl border border-slate-200 shadow-xs">
          <span className="text-xs text-slate-500 font-medium">Top Geographic Hub</span>
          <div className="text-2xl font-bold text-slate-900 mt-1">United States</div>
          <span className="text-[11px] text-slate-400">Followed by UK & Germany</span>
        </div>

        <div className="bg-white p-4 rounded-xl border border-slate-200 shadow-xs">
          <span className="text-xs text-slate-500 font-medium">Leading Professional Persona</span>
          <div className="text-2xl font-bold text-slate-900 mt-1">Tech & AI</div>
          <span className="text-[11px] text-emerald-600 font-medium">40.8% semantic match</span>
        </div>
      </div>

      {/* Charts Grid: Age Distribution & Professional Personas */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Age Brackets Bar Chart */}
        <div className="bg-white rounded-xl border border-slate-200 p-5 shadow-xs">
          <div className="flex items-center justify-between mb-4">
            <div>
              <h3 className="text-sm font-bold text-slate-900">Age Bracket Distribution</h3>
              <p className="text-xs text-slate-500">Multi-stage classifier: keywords, bio vectors & post scoring</p>
            </div>
            <span className="text-xs font-medium text-slate-500 bg-slate-100 px-2 py-0.5 rounded">
              6 Standard Cohorts
            </span>
          </div>

          <div className="h-56 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={ageData}>
                <XAxis dataKey="bracket" stroke="#94a3b8" fontSize={11} tickLine={false} />
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
                <Bar dataKey="count" fill="#6366f1" radius={[4, 4, 0, 0]} name="Users" />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Professional Personas */}
        <div className="bg-white rounded-xl border border-slate-200 p-5 shadow-xs flex flex-col justify-between">
          <div className="flex items-center justify-between mb-3">
            <div>
              <h3 className="text-sm font-bold text-slate-900">Professional Personas & Interests</h3>
              <p className="text-xs text-slate-500">Semantic zero-shot persona clustering via MiniLM embeddings</p>
            </div>
            <span className="text-xs font-semibold text-indigo-700 bg-indigo-50 px-2 py-0.5 rounded">
              Interest Taxonomies
            </span>
          </div>

          <div className="space-y-3">
            {personaData.map((item, idx) => (
              <div key={item.persona}>
                <div className="flex justify-between text-xs mb-1">
                  <span className="font-semibold text-slate-800">{item.persona}</span>
                  <span className="text-slate-500 font-medium">
                    {item.count.toLocaleString()} ({item.percentage}%)
                  </span>
                </div>
                <div className="w-full bg-slate-100 h-2 rounded-full overflow-hidden">
                  <div
                    className="h-full rounded-full transition-all"
                    style={{
                      width: `${item.percentage * 2}%`,
                      backgroundColor: COLORS[idx % COLORS.length],
                    }}
                  />
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Geography & Languages Breakdown */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Top Countries & Cities */}
        <div className="bg-white rounded-xl border border-slate-200 p-5 shadow-xs">
          <div className="flex items-center gap-2 mb-3">
            <Globe className="w-4 h-4 text-indigo-600" />
            <h3 className="text-sm font-bold text-slate-900">Geographic Distribution</h3>
          </div>
          <p className="text-xs text-slate-500 mb-4">
            spaCy NER location normalization against major world cities and gazetteers
          </p>

          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
            <div>
              <div className="text-[11px] font-bold uppercase tracking-wider text-slate-400 mb-2">
                Top Countries
              </div>
              <div className="space-y-2">
                {mockDemographics.top_countries.map((c) => (
                  <div key={c.country} className="flex items-center justify-between text-xs py-1 border-b border-slate-50">
                    <span className="font-medium text-slate-700">{c.country}</span>
                    <span className="font-bold text-slate-800">{c.percentage}%</span>
                  </div>
                ))}
              </div>
            </div>

            <div>
              <div className="text-[11px] font-bold uppercase tracking-wider text-slate-400 mb-2">
                Major Metro Hubs
              </div>
              <div className="space-y-2">
                {mockDemographics.top_cities.map((city) => (
                  <div key={city.city} className="flex items-center justify-between text-xs py-1 border-b border-slate-50">
                    <span className="font-medium text-slate-700 flex items-center gap-1">
                      <MapPin className="w-3 h-3 text-slate-400" />
                      {city.city}
                    </span>
                    <span className="font-bold text-slate-800">{city.percentage}%</span>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>

        {/* Top Languages */}
        <div className="bg-white rounded-xl border border-slate-200 p-5 shadow-xs">
          <div className="flex items-center gap-2 mb-3">
            <Languages className="w-4 h-4 text-indigo-600" />
            <h3 className="text-sm font-bold text-slate-900">Language Distribution</h3>
          </div>
          <p className="text-xs text-slate-500 mb-4">
            Language identification across posts and user descriptions
          </p>

          <div className="space-y-3">
            {mockDemographics.top_languages.map((l, idx) => (
              <div key={l.language}>
                <div className="flex justify-between text-xs mb-1">
                  <span className="font-semibold text-slate-800">{l.language}</span>
                  <span className="text-slate-500 font-medium">
                    {l.count.toLocaleString()} ({l.percentage}%)
                  </span>
                </div>
                <div className="w-full bg-slate-100 h-2 rounded-full overflow-hidden">
                  <div
                    className="h-full rounded-full transition-all"
                    style={{
                      width: `${l.percentage}%`,
                      backgroundColor: COLORS[idx % COLORS.length],
                    }}
                  />
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Privacy-Preserving Influencer Audience Inspector */}
      <div className="bg-white rounded-xl border border-slate-200 overflow-hidden shadow-xs">
        <div className="p-5 bg-slate-50 border-b border-slate-200 flex flex-wrap items-center justify-between gap-3">
          <div>
            <div className="flex items-center gap-2">
              <ShieldCheck className="w-4 h-4 text-emerald-600" />
              <h3 className="text-sm font-bold text-slate-900">
                Privacy-Preserving Follower Audience Inspector
              </h3>
            </div>
            <p className="text-xs text-slate-500 mt-0.5">
              Inspect an influencer&apos;s follower demographics with enforced k-anonymity privacy thresholds.
            </p>
          </div>

          <div className="flex items-center gap-3">
            <div className="text-xs text-slate-600">
              Privacy Threshold: <strong className="text-slate-900 font-bold">k ≥ {minGroupSize}</strong>
            </div>
            <StatusBadge type="privacy" value="Active" />
          </div>
        </div>

        <div className="p-5 space-y-4">
          <div className="flex items-center gap-3">
            <span className="text-xs font-medium text-slate-600">Auditing Influencer:</span>
            <div className="flex gap-2">
              {['alex_researcher', 'dr_elena_climate', 'marcus_macro'].map((id) => (
                <button
                  key={id}
                  type="button"
                  onClick={() => setSelectedInfluencer(id)}
                  className={`px-3 py-1 rounded-lg text-xs font-semibold transition-all ${
                    selectedInfluencer === id
                      ? 'bg-indigo-600 text-white shadow-2xs'
                      : 'bg-slate-100 hover:bg-slate-200 text-slate-700'
                  }`}
                >
                  @{id}
                </button>
              ))}
            </div>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 pt-2">
            {/* Audience Age Breakdown */}
            <div className="p-4 bg-slate-50 rounded-xl border border-slate-200">
              <span className="text-xs font-bold text-slate-800 mb-2 block">
                Follower Age Distribution
              </span>
              <div className="space-y-2">
                {Object.entries(mockInfluencerAudience.age_distribution).map(([age, count]) => (
                  <div key={age} className="flex justify-between text-xs py-1 border-b border-slate-100">
                    <span className="text-slate-600">{age}</span>
                    <span className="font-bold text-slate-800">{count.toLocaleString()}</span>
                  </div>
                ))}
              </div>
            </div>

            {/* Audience Top Personas */}
            <div className="p-4 bg-slate-50 rounded-xl border border-slate-200">
              <span className="text-xs font-bold text-slate-800 mb-2 block">
                Top Follower Personas
              </span>
              <div className="space-y-2">
                {mockInfluencerAudience.top_personas.map((p) => (
                  <div key={p.persona} className="flex justify-between text-xs py-1 border-b border-slate-100">
                    <span className="text-slate-600">{p.persona}</span>
                    <span className="font-bold text-slate-800">{p.percentage}%</span>
                  </div>
                ))}
              </div>
            </div>

            {/* Privacy Shield Explanation Card */}
            <div className="p-4 bg-emerald-50/50 rounded-xl border border-emerald-200 flex flex-col justify-between">
              <div>
                <span className="text-xs font-bold text-emerald-900 flex items-center gap-1.5 mb-1.5">
                  <ShieldCheck className="w-4 h-4 text-emerald-600" />
                  Privacy Protection Active
                </span>
                <p className="text-xs text-emerald-800/80 leading-relaxed">
                  All demographic segments with fewer than 5 members are automatically suppressed. Individual follower identities cannot be re-identified from this profile.
                </p>
              </div>
              <div className="text-[11px] text-emerald-700 font-medium pt-2 border-t border-emerald-200">
                Audited: {mockInfluencerAudience.total_followers_analyzed.toLocaleString()} followers
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
