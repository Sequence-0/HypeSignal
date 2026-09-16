import React from 'react';
import { Activity, RefreshCw, Radio, Database, CheckCircle2 } from 'lucide-react';
import { HealthResponse } from '../../types';

interface HeaderProps {
  health: HealthResponse | null;
  useMock: boolean;
  setUseMock: (val: boolean) => void;
  onRefresh: () => void;
  isLoading: boolean;
  sseConnected: boolean;
}

export const Header: React.FC<HeaderProps> = ({
  health,
  useMock,
  setUseMock,
  onRefresh,
  isLoading,
  sseConnected,
}) => {
  const isOnline = health?.status === 'ok';

  return (
    <header className="h-16 bg-white border-b border-slate-200 px-6 flex items-center justify-between sticky top-0 z-30 shadow-2xs">
      <div className="flex items-center gap-3">
        <div className="w-10 h-10 rounded-xl bg-gradient-to-tr from-indigo-600 to-violet-500 flex items-center justify-center text-white shadow-sm shadow-indigo-200">
          <Radio className="w-5 h-5 animate-pulse" />
        </div>
        <div>
          <div className="flex items-center gap-2">
            <h1 className="text-base font-bold text-slate-900 tracking-tight">HypeSignal</h1>
            <span className="px-2 py-0.5 text-[11px] font-semibold bg-indigo-50 text-indigo-700 border border-indigo-200 rounded-full">
              AI Analytics
            </span>
          </div>
          <p className="text-xs text-slate-500">Multi-Platform Audience Intelligence & Social Dynamics</p>
        </div>
      </div>

      <div className="flex items-center gap-4">
        {/* SSE Live Pulse */}
        <div className="hidden md:flex items-center gap-2 px-3 py-1.5 rounded-full bg-slate-50 border border-slate-200 text-xs text-slate-600">
          <span className="relative flex h-2 w-2">
            <span className={`animate-ping absolute inline-flex h-full w-full rounded-full ${sseConnected ? 'bg-emerald-400 opacity-75' : 'bg-amber-400 opacity-75'}`} />
            <span className={`relative inline-flex rounded-full h-2 w-2 ${sseConnected ? 'bg-emerald-500' : 'bg-amber-500'}`} />
          </span>
          <span className="font-medium text-slate-700">
            {sseConnected ? 'Live Stream Active' : 'Connecting to Stream...'}
          </span>
        </div>

        {/* Backend Status Pill */}
        <div className="hidden sm:flex items-center gap-2 px-3 py-1.5 rounded-full bg-slate-50 border border-slate-200 text-xs">
          <Database className="w-3.5 h-3.5 text-slate-500" />
          <span className="text-slate-600">Backend:</span>
          {isOnline ? (
            <span className="flex items-center gap-1 font-semibold text-emerald-600">
              <CheckCircle2 className="w-3.5 h-3.5" />
              Online (DuckDB)
            </span>
          ) : (
            <span className="font-semibold text-amber-600">Standby (Showcase)</span>
          )}
        </div>

        {/* Live / Demo Mode Selector */}
        <div className="flex items-center bg-slate-100 p-1 rounded-lg border border-slate-200 text-xs font-medium">
          <button
            type="button"
            onClick={() => setUseMock(false)}
            className={`px-3 py-1 rounded-md transition-all ${
              !useMock
                ? 'bg-white text-slate-900 shadow-2xs font-semibold'
                : 'text-slate-500 hover:text-slate-900'
            }`}
          >
            Live API
          </button>
          <button
            type="button"
            onClick={() => setUseMock(true)}
            className={`px-3 py-1 rounded-md transition-all ${
              useMock
                ? 'bg-white text-slate-900 shadow-2xs font-semibold'
                : 'text-slate-500 hover:text-slate-900'
            }`}
          >
            Showcase Mode
          </button>
        </div>

        {/* Refresh Button */}
        <button
          type="button"
          onClick={onRefresh}
          disabled={isLoading}
          aria-label="Refresh Dashboard Data"
          className="p-2 text-slate-500 hover:text-slate-800 bg-white hover:bg-slate-50 border border-slate-200 rounded-lg transition-colors disabled:opacity-50"
        >
          <RefreshCw className={`w-4 h-4 ${isLoading ? 'animate-spin text-indigo-600' : ''}`} />
        </button>
      </div>
    </header>
  );
};
