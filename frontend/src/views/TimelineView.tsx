import React, { useState } from 'react';
import {
  Clock,
  Search,
  Filter,
  GitBranch,
  Heart,
  Repeat,
  MessageCircle,
  Eye,
  ExternalLink,
  Calendar,
} from 'lucide-react';
import { CanonicalPost, ConversationThread } from '../types';
import { StatusBadge } from '../components/common/StatusBadge';
import { DiscussionTree } from '../components/visualizers/DiscussionTree';
import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
} from 'recharts';
import {
  mockConversationThread,
  mockPosts,
  mockTimeseries,
} from '../services/mockData';

interface TimelineViewProps {
  posts?: CanonicalPost[];
}

export const TimelineView: React.FC<TimelineViewProps> = ({ posts = mockPosts }) => {
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedPlatform, setSelectedPlatform] = useState<string>('all');
  const [activeThread, setActiveThread] = useState<ConversationThread | null>(null);

  const platforms = [
    { id: 'all', label: 'All Platforms' },
    { id: 'twitter', label: 'X (Twitter)' },
    { id: 'telegram', label: 'Telegram' },
    { id: 'youtube', label: 'YouTube' },
    { id: 'reddit', label: 'Reddit' },
    { id: 'bluesky', label: 'Bluesky' },
  ];

  const filteredPosts = posts.filter((p) => {
    const matchesPlatform =
      selectedPlatform === 'all' || p.platform.toLowerCase() === selectedPlatform;
    const matchesQuery =
      searchQuery === '' ||
      p.text.toLowerCase().includes(searchQuery.toLowerCase()) ||
      p.author_id.toLowerCase().includes(searchQuery.toLowerCase()) ||
      p.hashtags.some((h) => h.toLowerCase().includes(searchQuery.toLowerCase()));
    return matchesPlatform && matchesQuery;
  });

  return (
    <div className="space-y-6">
      {/* Top Section / Title */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div>
          <div className="flex items-center gap-2">
            <span className="px-2.5 py-0.5 rounded-full text-xs font-semibold bg-indigo-50 text-indigo-700 border border-indigo-200">
              Timeline & Dynamics
            </span>
            <h2 className="text-xl font-bold text-slate-900">
              Conversations & Discussion Dynamics
            </h2>
          </div>
          <p className="text-xs text-slate-500 mt-1">
            Chronological timeline with multi-platform search, filtering, and reply tree exploration.
          </p>
        </div>

        <div className="flex items-center gap-2 text-xs text-slate-500 bg-white px-3 py-1.5 rounded-lg border border-slate-200 shadow-2xs">
          <Calendar className="w-3.5 h-3.5 text-slate-400" />
          <span>Observation Window: <strong>Sep 01 - Sep 16, 2026</strong></span>
        </div>
      </div>

      {/* Activity Volume Over Time Chart */}
      <div className="bg-white rounded-xl border border-slate-200 p-5 shadow-xs">
        <div className="flex items-center justify-between mb-4">
          <div>
            <h3 className="text-sm font-bold text-slate-900">Conversation Volume Over Time</h3>
            <p className="text-xs text-slate-500">Historical post frequency grouped by 2-hour observation buckets</p>
          </div>
          <span className="text-xs font-medium text-slate-500 bg-slate-100 px-2.5 py-1 rounded-md">
            Interval: 2 Hours
          </span>
        </div>

        <div className="h-52 w-full">
          <ResponsiveContainer width="100%" height="100%">
            <AreaChart data={mockTimeseries}>
              <defs>
                <linearGradient id="timelineGrad" x1="0" y1="0" x2="0" y2="1">
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
                fill="url(#timelineGrad)"
              />
            </AreaChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Search & Platform Filter Bar */}
      <div className="bg-white rounded-xl border border-slate-200 p-4 shadow-xs space-y-3">
        <div className="flex flex-col sm:flex-row items-stretch sm:items-center gap-3">
          <div className="relative flex-1">
            <Search className="w-4 h-4 text-slate-400 absolute left-3 top-1/2 -translate-y-1/2" />
            <input
              type="text"
              placeholder="Search posts by keyword, author, or #hashtag..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="w-full pl-9 pr-4 py-2 bg-slate-50 border border-slate-200 rounded-lg text-xs text-slate-900 placeholder:text-slate-400 focus:outline-hidden focus:border-indigo-500 focus:bg-white transition-all"
            />
          </div>

          <div className="text-xs text-slate-500 font-medium px-2 shrink-0">
            Showing <strong>{filteredPosts.length}</strong> posts
          </div>
        </div>

        {/* Platform Pills */}
        <div className="flex flex-wrap items-center gap-2 pt-1 border-t border-slate-100">
          {platforms.map((plat) => (
            <button
              key={plat.id}
              type="button"
              onClick={() => setSelectedPlatform(plat.id)}
              className={`px-3 py-1.5 rounded-lg text-xs font-medium transition-all ${
                selectedPlatform === plat.id
                  ? 'bg-indigo-600 text-white shadow-2xs'
                  : 'bg-slate-100 hover:bg-slate-200 text-slate-600'
              }`}
            >
              {plat.label}
            </button>
          ))}
        </div>
      </div>

      {/* Discussion Tree Modal Overlay */}
      {activeThread && (
        <div className="fixed inset-0 z-50 bg-slate-900/40 backdrop-blur-xs flex items-center justify-center p-4 sm:p-6 overflow-y-auto">
          <div className="max-w-3xl w-full max-h-[90vh] overflow-y-auto">
            <DiscussionTree
              thread={activeThread}
              onClose={() => setActiveThread(null)}
            />
          </div>
        </div>
      )}

      {/* Chronological Posts Feed */}
      <div className="space-y-3">
        {filteredPosts.map((post) => (
          <div
            key={post.id}
            className="bg-white rounded-xl border border-slate-200 p-5 shadow-xs hover:border-slate-300 transition-all"
          >
            <div className="flex items-start justify-between gap-3 mb-2">
              <div className="flex items-center gap-2.5">
                <StatusBadge type="platform" value={post.platform} />
                <span className="font-bold text-xs text-slate-900">
                  @{post.author_screen_name || post.author_id}
                </span>
                <span className="text-[11px] text-slate-400">
                  {new Date(post.timestamp).toLocaleString([], {
                    month: 'short',
                    day: 'numeric',
                    hour: '2-digit',
                    minute: '2-digit',
                  })}
                </span>
              </div>

              {/* View Discussion Flow Button */}
              <button
                type="button"
                onClick={() => setActiveThread(mockConversationThread)}
                className="inline-flex items-center gap-1.5 px-3 py-1 text-xs font-semibold text-indigo-700 bg-indigo-50 hover:bg-indigo-100 border border-indigo-200 rounded-lg transition-colors"
              >
                <GitBranch className="w-3.5 h-3.5" />
                <span>View Discussion Flow</span>
              </button>
            </div>

            <p className="text-xs text-slate-800 leading-relaxed font-sans mt-2">
              {post.text}
            </p>

            {/* Hashtags */}
            {post.hashtags && post.hashtags.length > 0 && (
              <div className="flex flex-wrap gap-1.5 mt-3">
                {post.hashtags.map((tag) => (
                  <span
                    key={tag}
                    className="text-[11px] font-medium text-indigo-600 bg-indigo-50/60 px-2 py-0.5 rounded-md"
                  >
                    #{tag}
                  </span>
                ))}
              </div>
            )}

            {/* Engagement Metrics Footer */}
            <div className="mt-4 pt-3 border-t border-slate-100 flex flex-wrap items-center gap-5 text-xs text-slate-500">
              <div className="flex items-center gap-1.5">
                <Heart className="w-3.5 h-3.5 text-rose-500" />
                <span>{post.metrics.likes.toLocaleString()} likes</span>
              </div>
              <div className="flex items-center gap-1.5">
                <Repeat className="w-3.5 h-3.5 text-indigo-500" />
                <span>{post.metrics.reposts.toLocaleString()} reposts</span>
              </div>
              <div className="flex items-center gap-1.5">
                <MessageCircle className="w-3.5 h-3.5 text-amber-500" />
                <span>{post.metrics.replies.toLocaleString()} replies</span>
              </div>
              {(post.metrics.views || post.metrics.impressions) && (
                <div className="flex items-center gap-1.5 text-slate-400">
                  <Eye className="w-3.5 h-3.5" />
                  <span>
                    {((post.metrics.views || post.metrics.impressions) || 0).toLocaleString()} views
                  </span>
                </div>
              )}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};
