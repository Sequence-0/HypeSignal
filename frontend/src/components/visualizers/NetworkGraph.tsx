import React, { useState } from 'react';
import { Share2, Users, Network, Info } from 'lucide-react';
import { BridgeKOL, KOLProfile } from '../../types';

interface NetworkGraphProps {
  kols: KOLProfile[];
  bridgeKols: BridgeKOL[];
}

interface GraphNode {
  id: string;
  name: string;
  x: number;
  y: number;
  r: number;
  community: number;
  isBridge: boolean;
  score: number;
  inDegree: number;
  betweenness: number;
}

const COMMUNITY_COLORS: Record<number, { fill: string; stroke: string; label: string }> = {
  1: { fill: '#6366f1', stroke: '#4338ca', label: 'Tech & AI' },
  2: { fill: '#10b981', stroke: '#047857', label: 'Climate & Science' },
  3: { fill: '#f59e0b', stroke: '#b45309', label: 'Finance & Markets' },
  4: { fill: '#ec4899', stroke: '#be185d', label: 'Media & Culture' },
};

export const NetworkGraph: React.FC<NetworkGraphProps> = ({ kols, bridgeKols }) => {
  const [selectedNode, setSelectedNode] = useState<GraphNode | null>(null);

  // Layout node positions in a circular multi-cluster map
  const bridgeIds = new Set(bridgeKols.map((b) => b.user_id));

  // Synthesize nodes
  const nodes: GraphNode[] = kols.map((kol, idx) => {
    const isBridge = bridgeIds.has(kol.user_id);
    const comm = kol.community_id || (idx % 3 + 1);
    
    // Cluster positioning
    const angle = (idx / Math.max(kols.length, 1)) * 2 * Math.PI;
    const clusterOffset = comm === 1 ? { x: -70, y: -40 } : comm === 2 ? { x: 70, y: -40 } : { x: 0, y: 70 };
    const rDist = isBridge ? 35 : 120; // Bridge nodes live closer to the center!
    const x = 300 + clusterOffset.x + Math.cos(angle) * rDist;
    const y = 200 + clusterOffset.y + Math.sin(angle) * rDist;
    const r = Math.max(12, Math.min(26, Math.round(kol.composite_influence_score * 24)));

    return {
      id: kol.user_id,
      name: kol.screen_name || kol.user_id,
      x,
      y,
      r,
      community: comm,
      isBridge,
      score: kol.composite_influence_score,
      inDegree: kol.in_degree,
      betweenness: kol.betweenness_centrality,
    };
  });

  // Synthesize edges between nodes in the same community, plus bridge connections
  const edges: Array<{ from: GraphNode; to: GraphNode; isCross: boolean }> = [];
  for (let i = 0; i < nodes.length; i++) {
    for (let j = i + 1; j < nodes.length; j++) {
      const a = nodes[i];
      const b = nodes[j];
      const isCross = a.community !== b.community;
      if (!isCross || a.isBridge || b.isBridge) {
        // Only draw relevant connections
        if (Math.hypot(a.x - b.x, a.y - b.y) < 180) {
          edges.push({ from: a, to: b, isCross });
        }
      }
    }
  }

  return (
    <div className="bg-white rounded-xl border border-slate-200 overflow-hidden shadow-xs">
      <div className="p-5 border-b border-slate-200 flex flex-wrap items-center justify-between gap-3">
        <div>
          <div className="flex items-center gap-2">
            <Network className="w-5 h-5 text-indigo-600" />
            <h3 className="text-sm font-bold text-slate-900">
              Interactive Follower Network Topology
            </h3>
          </div>
          <p className="text-xs text-slate-500 mt-0.5">
            Click any node to inspect authority rank, community cluster, and bridge connectivity
          </p>
        </div>

        {/* Legend */}
        <div className="flex flex-wrap items-center gap-3 text-xs">
          {Object.entries(COMMUNITY_COLORS).slice(0, 3).map(([id, col]) => (
            <div key={id} className="flex items-center gap-1.5 text-slate-600 font-medium">
              <span className="w-2.5 h-2.5 rounded-full" style={{ backgroundColor: col.fill }} />
              {col.label}
            </div>
          ))}
          <div className="flex items-center gap-1.5 text-slate-600 font-medium">
            <span className="w-2.5 h-2.5 rounded-full border-2 border-indigo-600 bg-white" />
            Bridge Connectors
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3">
        {/* SVG Canvas */}
        <div className="lg:col-span-2 relative bg-slate-50/60 p-4 flex items-center justify-center min-h-[380px] border-b lg:border-b-0 lg:border-r border-slate-200">
          <svg viewBox="0 0 600 400" className="w-full h-auto max-h-[380px]">
            {/* Draw Links */}
            {edges.map((e, idx) => (
              <line
                key={idx}
                x1={e.from.x}
                y1={e.from.y}
                x2={e.to.x}
                y2={e.to.y}
                stroke={e.isCross ? '#cbd5e1' : '#e2e8f0'}
                strokeWidth={e.isCross ? 1.5 : 1}
                strokeDasharray={e.isCross ? '3,3' : undefined}
              />
            ))}

            {/* Draw Nodes */}
            {nodes.map((node) => {
              const commColor = COMMUNITY_COLORS[node.community] || COMMUNITY_COLORS[1];
              const isSelected = selectedNode?.id === node.id;

              return (
                <g
                  key={node.id}
                  className="cursor-pointer transition-transform hover:scale-110"
                  onClick={() => setSelectedNode(node)}
                >
                  {/* Bridge Pulsing Ring */}
                  {node.isBridge && (
                    <circle
                      cx={node.x}
                      cy={node.y}
                      r={node.r + 7}
                      fill="none"
                      stroke="#6366f1"
                      strokeWidth="2"
                      strokeDasharray="4,4"
                      className="animate-spin origin-center"
                      style={{ transformOrigin: `${node.x}px ${node.y}px` }}
                    />
                  )}

                  {/* Main Node Circle */}
                  <circle
                    cx={node.x}
                    cy={node.y}
                    r={node.r}
                    fill={commColor.fill}
                    stroke={isSelected ? '#0f172a' : '#ffffff'}
                    strokeWidth={isSelected ? 3 : 2}
                    className="shadow-sm"
                  />

                  {/* Text Label */}
                  <text
                    x={node.x}
                    y={node.y + node.r + 12}
                    textAnchor="middle"
                    fill="#334155"
                    fontSize="9"
                    fontWeight="600"
                  >
                    @{node.name}
                  </text>
                </g>
              );
            })}
          </svg>
        </div>

        {/* Node Detail Inspector */}
        <div className="p-5 flex flex-col justify-between">
          <div>
            <div className="text-xs font-bold uppercase tracking-wider text-slate-400 mb-3 flex items-center gap-1.5">
              <Info className="w-3.5 h-3.5" /> Node Inspector
            </div>

            {selectedNode ? (
              <div className="space-y-4">
                <div className="p-3 bg-slate-50 rounded-xl border border-slate-200">
                  <div className="flex items-center justify-between">
                    <span className="font-bold text-sm text-slate-900">@{selectedNode.name}</span>
                    {selectedNode.isBridge && (
                      <span className="px-2 py-0.5 rounded-full text-[10px] font-bold bg-indigo-100 text-indigo-800">
                        Bridge Connector
                      </span>
                    )}
                  </div>
                  <div className="text-xs text-slate-500 mt-1">
                    Community: <span className="font-semibold text-slate-700">{COMMUNITY_COLORS[selectedNode.community]?.label}</span>
                  </div>
                </div>

                <div className="space-y-2 text-xs">
                  <div className="flex justify-between py-1.5 border-b border-slate-100">
                    <span className="text-slate-500">Influence Score</span>
                    <span className="font-bold text-indigo-600">{(selectedNode.score * 100).toFixed(0)} / 100</span>
                  </div>
                  <div className="flex justify-between py-1.5 border-b border-slate-100">
                    <span className="text-slate-500">Audience Reach (Followers)</span>
                    <span className="font-bold text-slate-800">{selectedNode.inDegree.toLocaleString()}</span>
                  </div>
                  <div className="flex justify-between py-1.5 border-b border-slate-100">
                    <span className="text-slate-500">Betweenness (Connectivity)</span>
                    <span className="font-bold text-slate-800">{selectedNode.betweenness.toFixed(3)}</span>
                  </div>
                </div>

                {selectedNode.isBridge && (
                  <div className="p-3 rounded-lg bg-indigo-50 border border-indigo-200 text-xs text-indigo-900 leading-relaxed">
                    🌟 <strong>High Bridge Potential:</strong> This influencer actively connects distinct audience clusters, facilitating cross-pollination of ideas.
                  </div>
                )}
              </div>
            ) : (
              <div className="text-center py-10 px-4 text-slate-400">
                <Users className="w-8 h-8 mx-auto mb-2 text-slate-300" />
                <p className="text-xs font-medium">Click on any node in the network to inspect authority and community metrics.</p>
              </div>
            )}
          </div>

          <div className="text-[11px] text-slate-400 border-t border-slate-100 pt-3 mt-4">
            Graph partitioned via Louvain modularity with PageRank centrality ranking.
          </div>
        </div>
      </div>
    </div>
  );
};
