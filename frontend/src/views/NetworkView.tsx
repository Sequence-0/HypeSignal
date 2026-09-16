import React, { useState } from 'react';
import {
  Share2,
  Users,
  Award,
  Zap,
  Layers,
  Network,
  GitBranch,
  Shield,
  ArrowRight,
} from 'lucide-react';
import { StatusBadge } from '../components/common/StatusBadge';
import { NetworkGraph } from '../components/visualizers/NetworkGraph';
import { CascadePath } from '../components/visualizers/CascadePath';
import {
  mockBridgeKOLs,
  mockDiffusionReport,
  mockKOLs,
  mockNetworkOverview,
} from '../services/mockData';

export const NetworkView: React.FC = () => {
  return (
    <div className="space-y-6">
      {/* Title */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div>
          <div className="flex items-center gap-2">
            <span className="px-2.5 py-0.5 rounded-full text-xs font-semibold bg-amber-50 text-amber-700 border border-amber-200">
              Network & Influencers
            </span>
            <h2 className="text-xl font-bold text-slate-900">
              Influencer Networks & Viral Diffusion
            </h2>
          </div>
          <p className="text-xs text-slate-500 mt-1">
            Community connections, key opinion leader rankings, boundary-spanning bridge influencers, and cross-community message spread.
          </p>
        </div>

        <div className="flex items-center gap-2">
          <span className="px-3 py-1 bg-white border border-slate-200 rounded-lg text-xs font-semibold text-slate-700 shadow-2xs">
            Live Network Topology
          </span>
        </div>
      </div>

      {/* Network Overview Summary Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-4 gap-4">
        <div className="bg-white p-4 rounded-xl border border-slate-200 shadow-xs">
          <span className="text-xs text-slate-500 font-medium">Follower Graph Nodes</span>
          <div className="text-2xl font-bold text-slate-900 mt-1">
            {mockNetworkOverview.total_nodes.toLocaleString()}
          </div>
          <span className="text-[11px] text-slate-400">Identified unique users</span>
        </div>

        <div className="bg-white p-4 rounded-xl border border-slate-200 shadow-xs">
          <span className="text-xs text-slate-500 font-medium">Follower Connections (Edges)</span>
          <div className="text-2xl font-bold text-slate-900 mt-1">
            {mockNetworkOverview.total_edges.toLocaleString()}
          </div>
          <span className="text-[11px] text-slate-400">Directed relationship links</span>
        </div>

        <div className="bg-white p-4 rounded-xl border border-slate-200 shadow-xs">
          <span className="text-xs text-slate-500 font-medium">Network Density</span>
          <div className="text-2xl font-bold text-slate-900 mt-1">
            {(mockNetworkOverview.density * 1000).toFixed(2)}‰
          </div>
          <span className="text-[11px] text-slate-400">Interconnected density</span>
        </div>

        <div className="bg-white p-4 rounded-xl border border-slate-200 shadow-xs">
          <span className="text-xs text-slate-500 font-medium">Community Clusters</span>
          <div className="text-2xl font-bold text-slate-900 mt-1">
            {mockNetworkOverview.community_count} Clusters
          </div>
          <span className="text-[11px] text-indigo-600 font-medium">Distinct audience clusters</span>
        </div>
      </div>

      {/* Interactive Topology Visualizer */}
      <NetworkGraph kols={mockKOLs} bridgeKols={mockBridgeKOLs} />

      {/* Bridge Influencers (Boundary Spanners) Highlight */}
      <div className="bg-white rounded-xl border border-slate-200 p-5 shadow-xs">
        <div className="flex items-center justify-between mb-4">
          <div>
            <div className="flex items-center gap-2">
              <Zap className="w-5 h-5 text-indigo-600" />
              <h3 className="text-sm font-bold text-slate-900">
                Community Connectors (Boundary-Spanning Bridge Influencers)
              </h3>
            </div>
            <p className="text-xs text-slate-500 mt-0.5">
              Influencers with high betweenness centrality whose connections bridge disparate audience clusters.
            </p>
          </div>
          <span className="text-xs font-semibold px-2.5 py-1 bg-indigo-50 text-indigo-700 border border-indigo-200 rounded-full">
            Key Catalysts for Viral Spread
          </span>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {mockBridgeKOLs.map((bridge) => (
            <div
              key={bridge.user_id}
              className="p-4 bg-indigo-50/40 rounded-xl border border-indigo-200 flex flex-col justify-between"
            >
              <div>
                <div className="flex items-center justify-between mb-1.5">
                  <span className="font-bold text-xs text-slate-900">@{bridge.screen_name || bridge.user_id}</span>
                  <span className="text-[10px] font-bold px-2 py-0.5 rounded-full bg-indigo-600 text-white">
                    {(bridge.cross_community_ratio * 100).toFixed(0)}% Cross-Cluster Ratio
                  </span>
                </div>
                <p className="text-xs text-slate-600 leading-relaxed">
                  Bridges <strong>Community #{bridge.home_community}</strong> with{' '}
                  {bridge.connected_communities.filter((c) => c !== bridge.home_community).length} other distinct interest groups ({bridge.total_neighbors} direct active connections).
                </p>
              </div>

              <div className="mt-3 pt-2.5 border-t border-indigo-100 flex items-center justify-between text-xs">
                <span className="text-slate-500">Betweenness Centrality:</span>
                <span className="font-bold text-indigo-950">{bridge.betweenness_centrality.toFixed(3)}</span>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Key Opinion Leaders (KOLs) Leaderboard */}
      <div className="bg-white rounded-xl border border-slate-200 overflow-hidden shadow-xs">
        <div className="p-5 bg-slate-50 border-b border-slate-200 flex items-center justify-between">
          <div>
            <div className="flex items-center gap-2">
              <Award className="w-5 h-5 text-amber-500" />
              <h3 className="text-sm font-bold text-slate-900">
                Key Opinion Leaders (Influencer Leaderboard)
              </h3>
            </div>
            <p className="text-xs text-slate-500 mt-0.5">
              Ranked by composite influence score (PageRank, follower reach, and connectivity)
            </p>
          </div>
          <span className="text-xs font-semibold px-2.5 py-1 bg-amber-50 text-amber-700 border border-amber-200 rounded-full">
            Top 10 Opinion Leaders
          </span>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs">
            <thead className="bg-slate-50/70 border-b border-slate-200 text-slate-500 font-semibold uppercase tracking-wider text-[11px]">
              <tr>
                <th className="py-3 px-4">Rank & Handle</th>
                <th className="py-3 px-4">Influence Score</th>
                <th className="py-3 px-4">Follower Reach</th>
                <th className="py-3 px-4">PageRank</th>
                <th className="py-3 px-4">Community Cluster</th>
                <th className="py-3 px-4 text-right">Role</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100">
              {mockKOLs.map((kol, idx) => (
                <tr key={kol.user_id} className="hover:bg-slate-50/80 transition-colors">
                  <td className="py-3 px-4 font-bold text-slate-900 flex items-center gap-2">
                    <span className="w-5 text-slate-400 font-mono text-[11px]">#{idx + 1}</span>
                    @{kol.screen_name || kol.user_id}
                  </td>
                  <td className="py-3 px-4">
                    <span className="font-bold text-indigo-600 bg-indigo-50 px-2 py-0.5 rounded border border-indigo-200">
                      {(kol.composite_influence_score * 100).toFixed(0)} / 100
                    </span>
                  </td>
                  <td className="py-3 px-4 font-semibold text-slate-800">
                    {kol.in_degree.toLocaleString()} followers
                  </td>
                  <td className="py-3 px-4 text-slate-500 font-mono">
                    {kol.pagerank.toFixed(4)}
                  </td>
                  <td className="py-3 px-4">
                    <span className="px-2 py-0.5 bg-slate-100 rounded text-[11px] font-medium text-slate-700">
                      Community #{kol.community_id}
                    </span>
                  </td>
                  <td className="py-3 px-4 text-right">
                    {kol.betweenness_centrality > 0.05 ? (
                      <span className="text-[10px] font-bold px-2 py-0.5 bg-indigo-100 text-indigo-800 rounded-full">
                        Bridge Connector
                      </span>
                    ) : (
                      <span className="text-[10px] font-medium text-slate-400">
                        Domain Pillar
                      </span>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Cross-Segment Viral Cascade Diffusion Tracer */}
      <CascadePath report={mockDiffusionReport} />
    </div>
  );
};
