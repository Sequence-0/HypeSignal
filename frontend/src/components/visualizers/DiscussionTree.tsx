import React from 'react';
import { GitBranch, Clock, Users, Flame, MessageSquare, CornerDownRight } from 'lucide-react';
import { ConversationThread, TreeNode } from '../../types';
import { StatusBadge } from '../common/StatusBadge';

interface DiscussionTreeProps {
  thread: ConversationThread;
  onClose?: () => void;
}

const RenderNode: React.FC<{ node: TreeNode; isRoot?: boolean }> = ({ node, isRoot = false }) => {
  return (
    <div className={`relative ${!isRoot ? 'ml-6 pl-4 border-l-2 border-indigo-100 my-3' : 'mb-4'}`}>
      {!isRoot && (
        <div className="absolute -left-[2px] top-4 w-3.5 border-b-2 border-indigo-100" />
      )}
      <div className={`p-4 rounded-xl border transition-all ${
        isRoot
          ? 'bg-indigo-50/40 border-indigo-200 shadow-xs'
          : 'bg-white border-slate-200 hover:border-slate-300'
      }`}>
        <div className="flex items-center justify-between mb-1.5">
          <div className="flex items-center gap-2">
            <span className="font-bold text-xs text-slate-900">@{node.author_id}</span>
            {isRoot && (
              <span className="px-2 py-0.5 rounded text-[10px] font-bold bg-indigo-600 text-white">
                Original Post
              </span>
            )}
            <span className="text-[11px] text-slate-400">
              Depth: {node.depth}
            </span>
          </div>
          {node.sentiment && (
            <StatusBadge type="sentiment" value={node.sentiment} />
          )}
        </div>

        <p className="text-xs text-slate-700 leading-relaxed font-sans">{node.text}</p>

        <div className="mt-2 flex items-center justify-between text-[11px] text-slate-400 pt-2 border-t border-slate-100">
          <span className="flex items-center gap-1">
            <Clock className="w-3 h-3" />
            {new Date(node.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
          </span>
          {node.replies && node.replies.length > 0 && (
            <span className="text-indigo-600 font-medium flex items-center gap-1">
              <CornerDownRight className="w-3 h-3" />
              {node.replies.length} {node.replies.length === 1 ? 'reply' : 'replies'}
            </span>
          )}
        </div>
      </div>

      {node.replies && node.replies.length > 0 && (
        <div className="space-y-1">
          {node.replies.map((reply) => (
            <RenderNode key={reply.post_id} node={reply} />
          ))}
        </div>
      )}
    </div>
  );
};

export const DiscussionTree: React.FC<DiscussionTreeProps> = ({ thread, onClose }) => {
  const { metrics, sentiment_dynamics, thread_tree } = thread;
  const replySpeedMins = Math.round((metrics.mean_reply_latency_seconds || 480) / 60);

  return (
    <div className="bg-white rounded-2xl border border-slate-200 overflow-hidden shadow-md">
      {/* Header */}
      <div className="px-6 py-4 bg-slate-50 border-b border-slate-200 flex items-center justify-between">
        <div>
          <div className="flex items-center gap-2">
            <GitBranch className="w-5 h-5 text-indigo-600" />
            <h2 className="text-base font-bold text-slate-900">
              Discussion Flow & Reply Hierarchy
            </h2>
          </div>
          <p className="text-xs text-slate-500 mt-0.5">
            Full tree reconstruction showing how replies branched and emotions evolved
          </p>
        </div>
        {onClose && (
          <button
            type="button"
            onClick={onClose}
            className="text-xs px-3 py-1.5 rounded-lg border border-slate-300 text-slate-600 hover:bg-slate-100"
          >
            Close
          </button>
        )}
      </div>

      {/* Metrics Grid */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-4 p-6 bg-slate-50/50 border-b border-slate-200">
        <div className="p-3 bg-white rounded-xl border border-slate-200">
          <div className="flex items-center gap-1.5 text-xs text-slate-500 mb-1">
            <MessageSquare className="w-3.5 h-3.5 text-indigo-500" />
            <span>Total Replies</span>
          </div>
          <div className="text-xl font-bold text-slate-900">{metrics.total_replies}</div>
          <div className="text-[10px] text-slate-400 mt-0.5">across {metrics.participant_count} authors</div>
        </div>

        <div className="p-3 bg-white rounded-xl border border-slate-200">
          <div className="flex items-center gap-1.5 text-xs text-slate-500 mb-1">
            <GitBranch className="w-3.5 h-3.5 text-violet-500" />
            <span>Discussion Depth</span>
          </div>
          <div className="text-xl font-bold text-slate-900">{metrics.max_depth} levels</div>
          <div className="text-[10px] text-slate-400 mt-0.5">avg branching: {metrics.avg_branching_factor.toFixed(1)}x</div>
        </div>

        <div className="p-3 bg-white rounded-xl border border-slate-200">
          <div className="flex items-center gap-1.5 text-xs text-slate-500 mb-1">
            <Clock className="w-3.5 h-3.5 text-emerald-500" />
            <span>Avg Reply Speed</span>
          </div>
          <div className="text-xl font-bold text-slate-900">{replySpeedMins} mins</div>
          <div className="text-[10px] text-slate-400 mt-0.5">from parent post</div>
        </div>

        <div className="p-3 bg-white rounded-xl border border-slate-200">
          <div className="flex items-center gap-1.5 text-xs text-slate-500 mb-1">
            <Flame className="w-3.5 h-3.5 text-amber-500" />
            <span>Controversy Index</span>
          </div>
          <div className="text-xl font-bold text-slate-900">
            {((sentiment_dynamics.controversy_index || 0.18) * 100).toFixed(0)}%
          </div>
          <div className="text-[10px] text-slate-400 mt-0.5">
            supportive: {Math.round((sentiment_dynamics.supportive_ratio || 0.7) * 100)}%
          </div>
        </div>
      </div>

      {/* Discussion Tree Render */}
      <div className="p-6 max-h-[500px] overflow-y-auto">
        <RenderNode node={thread_tree} isRoot={true} />
      </div>
    </div>
  );
};
