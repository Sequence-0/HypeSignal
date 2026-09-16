import React, { useState, useEffect } from 'react';
import { Header } from './components/layout/Header';
import { Sidebar, NavTab } from './components/layout/Sidebar';
import { OverviewView } from './views/OverviewView';
import { TimelineView } from './views/TimelineView';
import { SentimentView } from './views/SentimentView';
import { DemographicsView } from './views/DemographicsView';
import { TrendsView } from './views/TrendsView';
import { NetworkView } from './views/NetworkView';
import { ConnectorsView } from './views/ConnectorsView';
import { api } from './services/api';
import { HealthResponse } from './types';

export function App() {
  const [activeTab, setActiveTab] = useState<NavTab>('overview');
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [useMock, setUseMock] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [sseConnected, setSseConnected] = useState(false);

  const fetchStatus = async () => {
    setIsLoading(true);
    try {
      const h = await api.getHealth();
      setHealth(h);
      if (h.status !== 'ok') {
        setUseMock(true);
      }
    } catch {
      setUseMock(true);
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    fetchStatus();

    // Check SSE stream availability
    let sse: EventSource | null = null;
    try {
      sse = new EventSource('/api/v1/streaming/events');
      sse.onopen = () => setSseConnected(true);
      sse.onerror = () => setSseConnected(false);
    } catch {
      setSseConnected(false);
    }

    return () => {
      if (sse) sse.close();
    };
  }, []);

  return (
    <div className="min-h-screen bg-slate-50 text-slate-900 flex flex-col font-sans">
      {/* Top Fixed Header */}
      <Header
        health={health}
        useMock={useMock}
        setUseMock={setUseMock}
        onRefresh={fetchStatus}
        isLoading={isLoading}
        sseConnected={sseConnected}
      />

      {/* Main App Body (Sidebar + View Canvas) */}
      <div className="flex-1 flex w-full">
        {/* Left Fixed Navigation Menu */}
        <Sidebar activeTab={activeTab} setActiveTab={setActiveTab} />

        {/* Right Scrollable Analytics View Canvas */}
        <main className="flex-1 p-6 md:p-8 max-w-7xl mx-auto w-full overflow-y-auto">
          {activeTab === 'overview' && <OverviewView setActiveTab={setActiveTab} />}
          {activeTab === 'timeline' && <TimelineView />}
          {activeTab === 'sentiment' && <SentimentView />}
          {activeTab === 'demographics' && <DemographicsView />}
          {activeTab === 'trends' && <TrendsView />}
          {activeTab === 'network' && <NetworkView />}
          {activeTab === 'connectors' && <ConnectorsView />}
        </main>
      </div>
    </div>
  );
}

export default App;
