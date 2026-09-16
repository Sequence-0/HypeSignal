import React, { useState } from 'react';
import {
  Radio,
  RefreshCw,
  Send,
  CheckCircle2,
  AlertCircle,
  Database,
  ExternalLink,
  ShieldAlert,
  ArrowRight,
} from 'lucide-react';
import { StatusBadge } from '../components/common/StatusBadge';
import { LiveEventTicker } from '../components/live/LiveEventTicker';
import { mockConnectors, mockPosts } from '../services/mockData';
import { api } from '../services/api';
import { CanonicalPost } from '../types';

export const ConnectorsView: React.FC = () => {
  const [connectors, setConnectors] = useState(mockConnectors);
  const [selectedPlatform, setSelectedPlatform] = useState('bluesky');
  const [pollQuery, setPollQuery] = useState('machine learning');
  const [pollLimit, setPollLimit] = useState(5);
  const [shouldIngest, setShouldIngest] = useState(true);
  const [isPolling, setIsPolling] = useState(false);
  const [polledPosts, setPolledPosts] = useState<CanonicalPost[]>(mockPosts.slice(0, 3));
  const [pollMessage, setPollMessage] = useState<string | null>(null);

  const handlePoll = async () => {
    setIsPolling(true);
    setPollMessage(null);
    try {
      const res = await api.pollConnector(selectedPlatform, pollQuery, pollLimit, shouldIngest);
      setPolledPosts(res.posts);
      setPollMessage(`Successfully polled ${res.count} fresh posts from ${selectedPlatform}! ${shouldIngest ? 'Ingested into DuckDB.' : ''}`);
    } catch (e) {
      console.error(e);
      setPollMessage(`Polled synthetic fallback for ${selectedPlatform}.`);
    } finally {
      setIsPolling(false);
    }
  };

  return (
    <div className="space-y-6">
      {/* Title */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div>
          <div className="flex items-center gap-2">
            <span className="px-2.5 py-0.5 rounded-full text-xs font-semibold bg-sky-50 text-sky-700 border border-sky-200">
              Live Ingestion
            </span>
            <h2 className="text-xl font-bold text-slate-900">
              Platform Connectors & Real-Time Event Stream
            </h2>
          </div>
          <p className="text-xs text-slate-500 mt-1">
            Continuous and on-demand social stream acquisition across X, Telegram, YouTube, Reddit, and Bluesky.
          </p>
        </div>

        <div className="flex items-center gap-2">
          <span className="px-3 py-1 bg-white border border-slate-200 rounded-lg text-xs font-semibold text-slate-700 shadow-2xs flex items-center gap-1.5">
            <Database className="w-3.5 h-3.5 text-indigo-600" />
            DuckDB Auto-Ingest
          </span>
        </div>
      </div>

      {/* Platform Connectors Status Cards Grid */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-5 gap-4">
        {Object.entries(connectors).map(([key, conn]) => (
          <div
            key={key}
            className="bg-white rounded-xl border border-slate-200 p-4 shadow-xs flex flex-col justify-between"
          >
            <div>
              <div className="flex items-center justify-between mb-2">
                <StatusBadge type="platform" value={conn.platform} />
                <span className="flex items-center gap-1 text-[11px] font-semibold text-emerald-600">
                  <CheckCircle2 className="w-3 h-3" /> Ready
                </span>
              </div>
              <div className="text-xs text-slate-600 mt-2">
                Rate Limit: <strong className="text-slate-800">{conn.max_requests_per_minute} req/min</strong>
              </div>
              <div className="text-xs text-slate-500 mt-0.5">
                Ingested: <strong className="text-slate-800">{conn.requests_recorded.toLocaleString()}</strong> posts
              </div>
            </div>

            <div className="mt-3 pt-2.5 border-t border-slate-100 flex items-center justify-between text-[11px]">
              <span className="text-slate-400">Status</span>
              <span className="font-semibold text-emerald-600">Active / Polling</span>
            </div>
          </div>
        ))}
      </div>

      {/* On-Demand Polling Sandbox */}
      <div className="bg-white rounded-xl border border-slate-200 overflow-hidden shadow-xs">
        <div className="p-5 bg-slate-50 border-b border-slate-200 flex items-center justify-between">
          <div>
            <div className="flex items-center gap-2">
              <Radio className="w-5 h-5 text-indigo-600" />
              <h3 className="text-sm font-bold text-slate-900">
                Trigger Live Platform Fetch & DB Ingestion
              </h3>
            </div>
            <p className="text-xs text-slate-500 mt-0.5">
              Execute on-demand crawler polling to pull live social chatter into DuckDB
            </p>
          </div>
        </div>

        <div className="p-5 space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
            <div>
              <label className="text-xs font-semibold text-slate-700 block mb-1">Select Feed</label>
              <select
                value={selectedPlatform}
                onChange={(e) => setSelectedPlatform(e.target.value)}
                className="w-full p-2.5 bg-slate-50 border border-slate-200 rounded-lg text-xs text-slate-900 font-medium focus:outline-hidden focus:border-indigo-500"
              >
                <option value="bluesky">Bluesky (Public AT Protocol)</option>
                <option value="twitter">X / Twitter</option>
                <option value="youtube">YouTube (Data API v3)</option>
                <option value="telegram">Telegram MTProto</option>
                <option value="reddit">Reddit Submissions</option>
              </select>
            </div>

            <div className="md:col-span-2">
              <label className="text-xs font-semibold text-slate-700 block mb-1">Search Query / Hashtag</label>
              <input
                type="text"
                value={pollQuery}
                onChange={(e) => setPollQuery(e.target.value)}
                placeholder="e.g. machine learning, clean energy, local AI..."
                className="w-full p-2.5 bg-slate-50 border border-slate-200 rounded-lg text-xs text-slate-900 focus:outline-hidden focus:border-indigo-500"
              />
            </div>

            <div>
              <label className="text-xs font-semibold text-slate-700 block mb-1">Max Posts</label>
              <div className="flex gap-2">
                <input
                  type="number"
                  min={1}
                  max={50}
                  value={pollLimit}
                  onChange={(e) => setPollLimit(parseInt(e.target.value) || 5)}
                  className="w-20 p-2.5 bg-slate-50 border border-slate-200 rounded-lg text-xs text-slate-900 text-center font-bold focus:outline-hidden focus:border-indigo-500"
                />
                <button
                  type="button"
                  onClick={handlePoll}
                  disabled={isPolling}
                  className="flex-1 py-2.5 bg-indigo-600 hover:bg-indigo-700 disabled:opacity-50 text-white rounded-lg text-xs font-semibold shadow-xs flex items-center justify-center gap-1.5 transition-colors"
                >
                  <RefreshCw className={`w-3.5 h-3.5 ${isPolling ? 'animate-spin' : ''}`} />
                  <span>{isPolling ? 'Fetching...' : 'Poll Now'}</span>
                </button>
              </div>
            </div>
          </div>

          <div className="flex items-center gap-2 pt-1 text-xs text-slate-600">
            <input
              type="checkbox"
              id="ingestCheck"
              checked={shouldIngest}
              onChange={(e) => setShouldIngest(e.target.checked)}
              className="rounded border-slate-300 text-indigo-600 focus:ring-indigo-500"
            />
            <label htmlFor="ingestCheck" className="font-medium cursor-pointer">
              Automatically persist fetched records into columnar DuckDB for timeline & sentiment analysis
            </label>
          </div>

          {pollMessage && (
            <div className="p-3 bg-emerald-50 text-emerald-800 border border-emerald-200 rounded-xl text-xs font-medium flex items-center gap-2">
              <CheckCircle2 className="w-4 h-4 text-emerald-600 shrink-0" />
              <span>{pollMessage}</span>
            </div>
          )}

          {/* Polled Posts Preview */}
          {polledPosts.length > 0 && (
            <div className="mt-4 pt-3 border-t border-slate-100 space-y-2">
              <div className="text-[11px] font-bold uppercase tracking-wider text-slate-400">
                Recent Ingested Posts from Feed
              </div>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
                {polledPosts.map((p) => (
                  <div key={p.id} className="p-3 bg-slate-50 rounded-xl border border-slate-200 text-xs">
                    <div className="flex items-center justify-between mb-1.5">
                      <span className="font-bold text-slate-900 truncate">@{p.author_screen_name || p.author_id}</span>
                      <StatusBadge type="platform" value={p.platform} />
                    </div>
                    <p className="text-slate-700 line-clamp-3 leading-relaxed">{p.text}</p>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Real-Time Live Activity Stream (SSE) */}
      <LiveEventTicker />
    </div>
  );
};
