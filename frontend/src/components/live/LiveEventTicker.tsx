import React, { useState, useEffect } from 'react';
import { Radio, AlertTriangle, MessageSquare, Terminal, Pause, Play, Sparkles } from 'lucide-react';
import { LiveStreamEvent } from '../../types';

interface LiveEventTickerProps {
  onEventReceived?: (event: LiveStreamEvent) => void;
}

export const LiveEventTicker: React.FC<LiveEventTickerProps> = ({ onEventReceived }) => {
  const [events, setEvents] = useState<LiveStreamEvent[]>([
    {
      id: 'init_1',
      type: 'system',
      title: 'Streaming Pipeline Connected',
      detail: 'EventBroadcaster listening for real-time connector ticks and alerts.',
      timestamp: new Date().toLocaleTimeString(),
    },
    {
      id: 'init_2',
      type: 'burst',
      title: 'Viral Burst Detected: #LocalAI',
      detail: 'Term frequency accelerated by +5.82σ above rolling baseline.',
      timestamp: new Date(Date.now() - 60000).toLocaleTimeString(),
    },
    {
      id: 'init_3',
      type: 'post',
      platform: 'bluesky',
      title: 'New Bluesky Post Ingested',
      detail: 'Author did:plc:sarah published update on data compliance frameworks.',
      timestamp: new Date(Date.now() - 120000).toLocaleTimeString(),
    },
  ]);

  const [isPaused, setIsPaused] = useState(false);
  const [filter, setFilter] = useState<'all' | 'burst' | 'post' | 'system'>('all');

  useEffect(() => {
    // Connect to SSE stream
    let eventSource: EventSource | null = null;
    try {
      eventSource = new EventSource('/api/v1/streaming/events');

      eventSource.onmessage = (e) => {
        try {
          const data = JSON.parse(e.data);
          const newEvt: LiveStreamEvent = {
            id: `evt_${Date.now()}_${Math.random()}`,
            type: data.type || (data.event === 'burst' ? 'burst' : 'post'),
            platform: data.platform,
            title: data.title || (data.event ? `Event: ${data.event}` : 'Pipeline Event'),
            detail: data.detail || data.message || JSON.stringify(data),
            timestamp: new Date().toLocaleTimeString(),
          };

          if (!isPaused) {
            setEvents((prev) => [newEvt, ...prev.slice(0, 49)]);
          }
          if (onEventReceived) {
            onEventReceived(newEvt);
          }
        } catch (err) {
          console.debug('Unparsed SSE event:', e.data);
        }
      };

      eventSource.onerror = () => {
        // Fallback simulation when stream drops
        console.debug('SSE disconnected or offline.');
      };
    } catch (e) {
      console.warn('SSE initialization skipped:', e);
    }

    return () => {
      if (eventSource) {
        eventSource.close();
      }
    };
  }, [isPaused, onEventReceived]);

  const filteredEvents = filter === 'all' ? events : events.filter((e) => e.type === filter);

  return (
    <div className="bg-white rounded-xl border border-slate-200 overflow-hidden shadow-xs">
      <div className="p-4 bg-slate-50 border-b border-slate-200 flex flex-wrap items-center justify-between gap-3">
        <div className="flex items-center gap-2">
          <div className="relative flex h-2.5 w-2.5">
            <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75" />
            <span className="relative inline-flex rounded-full h-2.5 w-2.5 bg-emerald-500" />
          </div>
          <span className="text-xs font-bold text-slate-800">Live Activity Stream (SSE)</span>
          <span className="text-[11px] text-slate-400">/api/v1/streaming/events</span>
        </div>

        <div className="flex items-center gap-2">
          {/* Filter Pills */}
          <div className="flex items-center bg-slate-200/70 p-0.5 rounded-lg text-[11px]">
            {(['all', 'burst', 'post', 'system'] as const).map((f) => (
              <button
                key={f}
                type="button"
                onClick={() => setFilter(f)}
                className={`px-2 py-0.5 rounded-md capitalize transition-all ${
                  filter === f
                    ? 'bg-white text-slate-900 font-semibold shadow-2xs'
                    : 'text-slate-600 hover:text-slate-900'
                }`}
              >
                {f}
              </button>
            ))}
          </div>

          <button
            type="button"
            onClick={() => setIsPaused(!isPaused)}
            title={isPaused ? 'Resume live updates' : 'Pause live updates'}
            className="p-1.5 rounded-md hover:bg-slate-200/80 text-slate-500 transition-colors"
          >
            {isPaused ? <Play className="w-3.5 h-3.5 text-emerald-600" /> : <Pause className="w-3.5 h-3.5" />}
          </button>
        </div>
      </div>

      <div className="divide-y divide-slate-100 max-h-72 overflow-y-auto font-sans">
        {filteredEvents.map((evt) => {
          const isBurst = evt.type === 'burst';
          const isPost = evt.type === 'post';

          return (
            <div key={evt.id} className="p-3.5 hover:bg-slate-50/70 transition-colors flex items-start gap-3">
              <div
                className={`p-1.5 rounded-lg mt-0.5 shrink-0 ${
                  isBurst
                    ? 'bg-rose-50 text-rose-600'
                    : isPost
                    ? 'bg-sky-50 text-sky-600'
                    : 'bg-slate-100 text-slate-600'
                }`}
              >
                {isBurst ? (
                  <AlertTriangle className="w-3.5 h-3.5" />
                ) : isPost ? (
                  <MessageSquare className="w-3.5 h-3.5" />
                ) : (
                  <Terminal className="w-3.5 h-3.5" />
                )}
              </div>

              <div className="flex-1 min-w-0">
                <div className="flex items-center justify-between gap-2">
                  <span className="text-xs font-semibold text-slate-900 truncate">
                    {evt.title}
                  </span>
                  <span className="text-[10px] text-slate-400 shrink-0 font-mono">
                    {evt.timestamp}
                  </span>
                </div>
                <p className="text-[11px] text-slate-500 mt-0.5 leading-normal">
                  {evt.detail}
                </p>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
};
