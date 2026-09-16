import React from 'react';
import {
  BarChart3,
  Clock,
  Smile,
  Users,
  TrendingUp,
  Share2,
  Radio,
  Sparkles,
  Shield,
} from 'lucide-react';

export type NavTab = 'overview' | 'timeline' | 'sentiment' | 'demographics' | 'trends' | 'network' | 'connectors';

interface SidebarProps {
  activeTab: NavTab;
  setActiveTab: (tab: NavTab) => void;
  burstCount?: number;
}

export const Sidebar: React.FC<SidebarProps> = ({
  activeTab,
  setActiveTab,
  burstCount = 4,
}) => {
  const menuItems: Array<{
    id: NavTab;
    label: string;
    subtitle: string;
    icon: React.ComponentType<{ className?: string }>;
    pillar: string;
    badge?: string | number;
    badgeColor?: string;
  }> = [
    {
      id: 'overview',
      label: 'Executive Pulse',
      subtitle: 'Cross-pillar summary',
      icon: BarChart3,
      pillar: 'Executive',
    },
    {
      id: 'timeline',
      label: 'Conversations & Timeline',
      subtitle: 'Time slices & reply trees',
      icon: Clock,
      pillar: 'Timeline',
    },
    {
      id: 'sentiment',
      label: 'Audience Feelings & Tone',
      subtitle: 'Nuanced emotions & sarcasm',
      icon: Smile,
      pillar: 'Sentiment',
    },
    {
      id: 'demographics',
      label: 'Audience & Demographics',
      subtitle: 'Age, map, languages, personas',
      icon: Users,
      pillar: 'Demographics',
    },
    {
      id: 'trends',
      label: 'Rising Trends & Virality',
      subtitle: 'Momentum & narrative shift',
      icon: TrendingUp,
      pillar: 'Trends',
      badge: burstCount > 0 ? `${burstCount} Spikes` : undefined,
      badgeColor: 'bg-rose-500 text-white',
    },
    {
      id: 'network',
      label: 'Network & Viral Spread',
      subtitle: 'Top influencers & cascades',
      icon: Share2,
      pillar: 'Network',
    },
    {
      id: 'connectors',
      label: 'Live Feeds & Sources',
      subtitle: 'X, Telegram, YT, Reddit, Bluesky',
      icon: Radio,
      pillar: 'Feeds',
    },
  ];

  return (
    <aside className="w-68 bg-white border-r border-slate-200 flex flex-col justify-between shrink-0 h-[calc(100vh-4rem)] sticky top-16 select-none">
      <div className="p-3 space-y-1 overflow-y-auto">
        <div className="px-3 pt-2 pb-1 text-[11px] font-semibold uppercase tracking-wider text-slate-400">
          Analytics Menus
        </div>

        {menuItems.map((item) => {
          const Icon = item.icon;
          const isActive = activeTab === item.id;
          return (
            <button
              key={item.id}
              type="button"
              onClick={() => setActiveTab(item.id)}
              className={`w-full flex items-start gap-3 px-3 py-2.5 rounded-xl transition-all text-left group ${
                isActive
                  ? 'bg-indigo-50/80 border border-indigo-200/80 text-indigo-950 shadow-2xs'
                  : 'text-slate-600 hover:bg-slate-50 hover:text-slate-900 border border-transparent'
              }`}
            >
              <div
                className={`p-2 rounded-lg transition-colors mt-0.5 ${
                  isActive
                    ? 'bg-indigo-600 text-white shadow-xs'
                    : 'bg-slate-100 text-slate-500 group-hover:bg-slate-200/70 group-hover:text-slate-800'
                }`}
              >
                <Icon className="w-4 h-4" />
              </div>

              <div className="flex-1 min-w-0">
                <div className="flex items-center justify-between">
                  <span className={`text-xs font-semibold truncate ${isActive ? 'text-indigo-950' : 'text-slate-800'}`}>
                    {item.label}
                  </span>
                  {item.badge && (
                    <span
                      className={`text-[10px] font-bold px-1.5 py-0.5 rounded-full ${
                        item.badgeColor || 'bg-indigo-100 text-indigo-700'
                      }`}
                    >
                      {item.badge}
                    </span>
                  )}
                </div>
                <p className="text-[11px] text-slate-400 truncate mt-0.5">
                  {item.subtitle}
                </p>
              </div>
            </button>
          );
        })}
      </div>

      {/* Footer Info Box */}
      <div className="p-4 border-t border-slate-200 bg-slate-50/70 m-3 rounded-xl border">
        <div className="flex items-center gap-2 mb-1.5">
          <Sparkles className="w-4 h-4 text-indigo-600" />
          <span className="text-xs font-bold text-slate-800">Social Intelligence</span>
        </div>
        <p className="text-[11px] text-slate-500 leading-relaxed">
          Real-time audience sentiment, viral trend detection, and multi-platform network flow.
        </p>
        <div className="mt-2.5 pt-2 border-t border-slate-200 flex items-center justify-between text-[10px] text-slate-400">
          <span className="inline-flex items-center gap-1 font-medium text-slate-600">
            <Shield className="w-3 h-3 text-emerald-600" /> Privacy Protected
          </span>
          <span>v0.1.0</span>
        </div>
      </div>
    </aside>
  );
};
