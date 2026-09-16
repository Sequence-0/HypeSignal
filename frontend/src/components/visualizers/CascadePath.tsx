import React from 'react';
import { GitCommit, ArrowRight, Clock, Award, Layers } from 'lucide-react';
import { CrossSegmentDiffusionReport } from '../../types';
import { StatusBadge } from '../common/StatusBadge';

interface CascadePathProps {
  report: CrossSegmentDiffusionReport;
}

export const CascadePath: React.FC<CascadePathProps> = ({ report }) => {
  const { adoption_timeline, structural_virality_wiener, propagation_depth, half_life_seconds } = report;

  return (
    <div className="bg-white rounded-xl border border-slate-200 p-5 shadow-xs">
      <div className="flex flex-wrap items-center justify-between gap-4 mb-6">
        <div>
          <div className="flex items-center gap-2">
            <Layers className="w-5 h-5 text-indigo-600" />
            <h3 className="text-sm font-bold text-slate-900">
              Cross-Segment Viral Diffusion Sequence
            </h3>
          </div>
          <p className="text-xs text-slate-500 mt-0.5">
            Traces how the narrative propagated across follower communities and demographic brackets
          </p>
        </div>

        {/* Structural Metrics */}
        <div className="flex items-center gap-3">
          <div className="px-3 py-1.5 bg-slate-50 rounded-lg border border-slate-200 text-xs">
            <span className="text-slate-500">Viral Breadth (Wiener):</span>{' '}
            <span className="font-bold text-slate-800">{structural_virality_wiener.toFixed(2)}</span>
          </div>
          <div className="px-3 py-1.5 bg-slate-50 rounded-lg border border-slate-200 text-xs">
            <span className="text-slate-500">Max Hop Depth:</span>{' '}
            <span className="font-bold text-slate-800">{propagation_depth} hops</span>
          </div>
          <div className="px-3 py-1.5 bg-slate-50 rounded-lg border border-slate-200 text-xs">
            <span className="text-slate-500">Half-Life:</span>{' '}
            <span className="font-bold text-slate-800">{Math.round(half_life_seconds / 60)} mins</span>
          </div>
        </div>
      </div>

      {/* Horizontal / Step-by-step Diffusion Timeline */}
      <div className="relative pl-6 space-y-6 before:absolute before:left-2 before:top-2 before:bottom-2 before:w-0.5 before:bg-indigo-100">
        {adoption_timeline.map((event, idx) => {
          const lagMins = Math.round(event.seconds_since_origin / 60);
          const isOrigin = idx === 0;

          return (
            <div key={idx} className="relative flex items-start gap-4 group">
              {/* Dot Icon */}
              <div
                className={`absolute -left-6 mt-1 w-4 h-4 rounded-full border-2 bg-white flex items-center justify-center transition-all ${
                  isOrigin
                    ? 'border-indigo-600 ring-4 ring-indigo-50'
                    : 'border-slate-300 group-hover:border-indigo-500'
                }`}
              >
                <div
                  className={`w-1.5 h-1.5 rounded-full ${isOrigin ? 'bg-indigo-600' : 'bg-slate-400'}`}
                />
              </div>

              {/* Step Card */}
              <div className="flex-1 bg-slate-50 hover:bg-white border border-slate-200 rounded-xl p-3.5 transition-all shadow-2xs hover:shadow-xs">
                <div className="flex flex-wrap items-center justify-between gap-2 mb-1">
                  <div className="flex items-center gap-2">
                    <span className="text-xs font-bold text-slate-900">{event.segment}</span>
                    {isOrigin && (
                      <span className="px-2 py-0.5 rounded text-[10px] font-bold bg-indigo-600 text-white">
                        Cascade Origin
                      </span>
                    )}
                  </div>
                  <div className="flex items-center gap-2 text-xs">
                    <span className="text-slate-500 flex items-center gap-1">
                      <Clock className="w-3 h-3" />
                      +{lagMins} mins
                    </span>
                    {event.effective_polarity && (
                      <StatusBadge type="sentiment" value={event.effective_polarity} />
                    )}
                  </div>
                </div>

                <div className="text-xs text-slate-600 flex items-center gap-2">
                  <span>Adopter:</span>
                  <span className="font-semibold text-slate-800">@{event.user_id}</span>
                  {event.sentiment_score !== undefined && (
                    <span className="text-[11px] text-slate-400">
                      (Sentiment score: {event.sentiment_score.toFixed(2)})
                    </span>
                  )}
                </div>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
};
