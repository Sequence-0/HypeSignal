import {
  ActivityTimeseriesPoint,
  AggregateDemographics,
  BridgeKOL,
  BurstAlert,
  CanonicalPost,
  ConnectorInfo,
  ConversationThread,
  CrossSegmentDiffusionReport,
  DynamicTopic,
  HealthResponse,
  InfluencerAudienceProfile,
  KOLProfile,
  NarrativeDriftAlert,
  SentimentAnalyzeResult,
  TemporalSentimentPoint,
  TimelineBounds,
  TrendForecast,
  UserProfile,
} from '../types';
import * as mock from './mockData';

const BASE_URL = '';

async function safeFetch<T>(url: string, options?: RequestInit, fallback?: T): Promise<T> {
  try {
    const res = await fetch(`${BASE_URL}${url}`, {
      ...options,
      headers: {
        'Content-Type': 'application/json',
        ...(options?.headers || {}),
      },
    });
    if (!res.ok) {
      throw new Error(`HTTP ${res.status}: ${res.statusText}`);
    }
    const data = await res.json();
    return data as T;
  } catch (err) {
    console.warn(`API request to ${url} failed, using showcase data.`, err);
    if (fallback !== undefined) {
      return fallback;
    }
    throw err;
  }
}

export const api = {
  // System Health
  async getHealth(): Promise<HealthResponse> {
    return safeFetch<HealthResponse>('/health', undefined, mock.mockHealth);
  },

  // Component A: Timeline
  async getTimelineBounds(): Promise<TimelineBounds> {
    return safeFetch<TimelineBounds>('/api/v1/timeline/bounds', undefined, mock.mockTimelineBounds);
  },

  async getTimelineSlice(params?: {
    start_time?: string;
    end_time?: string;
    platform?: string;
    keyword?: string;
    limit?: number;
  }): Promise<CanonicalPost[]> {
    const query = new URLSearchParams();
    if (params?.start_time) query.append('start_time', params.start_time);
    if (params?.end_time) query.append('end_time', params.end_time);
    if (params?.platform && params.platform !== 'all') query.append('platform', params.platform);
    if (params?.keyword) query.append('keyword', params.keyword);
    if (params?.limit) query.append('limit', params.limit.toString());

    const url = `/api/v1/timeline/slice?${query.toString()}`;
    const res = await safeFetch<CanonicalPost[]>(url, undefined, mock.mockPosts);
    return res.length > 0 ? res : mock.mockPosts;
  },

  async getActivityTimeseries(interval = '1 hour'): Promise<ActivityTimeseriesPoint[]> {
    const res = await safeFetch<{ points: ActivityTimeseriesPoint[] }>(
      `/api/v1/timeline/timeseries?interval=${encodeURIComponent(interval)}`,
      undefined,
      { points: mock.mockTimeseries }
    );
    return res.points && res.points.length > 0 ? res.points : mock.mockTimeseries;
  },

  async getConversationAnalytics(conversationId: string): Promise<ConversationThread> {
    return safeFetch<ConversationThread>(
      `/api/v1/analytics/conversations/${encodeURIComponent(conversationId)}`,
      undefined,
      mock.mockConversationThread
    );
  },

  // Component B: Sentiment
  async analyzeSentiment(text: string, stanceTarget?: string): Promise<SentimentAnalyzeResult> {
    try {
      const res = await fetch('/api/v1/sentiment/analyze', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text, stance_target: stanceTarget || null }),
      });
      if (res.ok) {
        const data = await res.json();
        if (data.results && data.results.length > 0) {
          const r = data.results[0];
          return {
            text: r.text || text,
            sentiment: r.sentiment?.polarity || 'neutral',
            sentiment_score: r.adjusted_sentiment_score ?? (r.sentiment?.score || 0),
            emotions: r.emotion?.probabilities || { joy: 0.5, optimism: 0.5 },
            primary_emotion: r.emotion?.primary_emotion || 'neutral',
            irony_score: r.irony?.irony_score || 0,
            is_ironic: Boolean(r.irony?.is_ironic),
            effective_polarity: r.effective_polarity || r.sentiment?.polarity || 'neutral',
            irony_inverted: Boolean(r.is_sarcasm_inverted),
            stance_target: stanceTarget,
            stance: r.stance?.stance || 'supportive',
            stance_score: r.stance?.score || 0.8,
          };
        }
      }
    } catch (e) {
      console.warn('Real-time sentiment analyze fallback triggered:', e);
    }

    // Local fallback sentiment simulation
    const isNegative = text.toLowerCase().includes('flaw') || text.toLowerCase().includes('fail') || text.toLowerCase().includes('bad') || text.toLowerCase().includes('critic');
    const isIronic = text.includes('🙄') || text.toLowerCase().includes('oh great') || text.toLowerCase().includes('groundbreaking');
    return {
      text,
      sentiment: isNegative ? 'negative' : 'positive',
      sentiment_score: isNegative ? -0.65 : 0.75,
      emotions: {
        joy: isNegative ? 0.05 : 0.65,
        optimism: isNegative ? 0.08 : 0.72,
        anger: isNegative ? 0.45 : 0.02,
        sadness: isNegative ? 0.35 : 0.01,
        fear: isNegative ? 0.25 : 0.03,
        surprise: isIronic ? 0.60 : 0.30,
        disgust: isNegative ? 0.40 : 0.02,
        anxiety: isNegative ? 0.55 : 0.12,
        excitement: isNegative ? 0.05 : 0.80,
      },
      primary_emotion: isIronic ? 'surprise' : (isNegative ? 'anxiety' : 'optimism'),
      irony_score: isIronic ? 0.88 : 0.12,
      is_ironic: isIronic,
      effective_polarity: isIronic ? 'negative' : (isNegative ? 'negative' : 'positive'),
      irony_inverted: isIronic,
      stance_target: stanceTarget,
      stance: isNegative ? 'against' : 'supportive',
      stance_score: isNegative ? -0.7 : 0.85,
    };
  },

  async getTemporalSentiment(): Promise<TemporalSentimentPoint[]> {
    const res = await safeFetch<{ points: TemporalSentimentPoint[] }>(
      '/api/v1/sentiment/temporal',
      undefined,
      { points: mock.mockTemporalSentiment }
    );
    return res.points && res.points.length > 0 ? res.points : mock.mockTemporalSentiment;
  },

  // Component C: Demographics
  async getDemographicsBreakdown(): Promise<AggregateDemographics> {
    const res = await safeFetch<AggregateDemographics>(
      '/api/v1/demographics/breakdown',
      undefined,
      mock.mockDemographics
    );
    return (res.total_users_profiled && res.total_users_profiled > 0) ? res : mock.mockDemographics;
  },

  async getProfiledUsers(): Promise<UserProfile[]> {
    const res = await safeFetch<UserProfile[]>(
      '/api/v1/demographics/users',
      undefined,
      mock.mockProfiledUsers
    );
    return res.length > 0 ? res : mock.mockProfiledUsers;
  },

  async getInfluencerAudience(influencerId: string, minGroupSize = 5): Promise<InfluencerAudienceProfile> {
    return safeFetch<InfluencerAudienceProfile>(
      `/api/v1/analytics/audience/${encodeURIComponent(influencerId)}?min_group_size=${minGroupSize}`,
      undefined,
      mock.mockInfluencerAudience
    );
  },

  // Component D: Trends
  async getBurstAlerts(): Promise<BurstAlert[]> {
    const res = await safeFetch<BurstAlert[]>(
      '/api/v1/trends/bursts?only_active=false',
      undefined,
      mock.mockBurstAlerts
    );
    return res.length > 0 ? res : mock.mockBurstAlerts;
  },

  async getDynamicTopics(): Promise<DynamicTopic[]> {
    const res = await safeFetch<{ topics: DynamicTopic[] }>(
      '/api/v1/trends/topics',
      undefined,
      { topics: mock.mockDynamicTopics }
    );
    return res.topics && res.topics.length > 0 ? res.topics : mock.mockDynamicTopics;
  },

  async getTrendForecasts(): Promise<TrendForecast[]> {
    const res = await safeFetch<TrendForecast[]>(
      '/api/v1/analytics/trends/forecast',
      undefined,
      mock.mockTrendForecasts
    );
    return res.length > 0 ? res : mock.mockTrendForecasts;
  },

  async getNarrativeDrift(term = 'Quantum Computing'): Promise<NarrativeDriftAlert> {
    return safeFetch<NarrativeDriftAlert>(
      `/api/v1/analytics/trends/narrative-drift?term=${encodeURIComponent(term)}`,
      undefined,
      mock.mockNarrativeDrift
    );
  },

  // Component E: Network
  async getNetworkOverview(): Promise<typeof mock.mockNetworkOverview> {
    const res = await safeFetch<typeof mock.mockNetworkOverview>(
      '/api/v1/network/overview',
      undefined,
      mock.mockNetworkOverview
    );
    return (res.total_nodes && res.total_nodes > 0) ? res : mock.mockNetworkOverview;
  },

  async getTopKOLs(topK = 10): Promise<KOLProfile[]> {
    const res = await safeFetch<KOLProfile[]>(
      `/api/v1/network/kols?top_k=${topK}`,
      undefined,
      mock.mockKOLs
    );
    return res.length > 0 ? res : mock.mockKOLs;
  },

  async getBridgeKOLs(): Promise<BridgeKOL[]> {
    const res = await safeFetch<{ leaders: BridgeKOL[] }>(
      '/api/v1/analytics/influencers/bridge-kols',
      undefined,
      { leaders: mock.mockBridgeKOLs }
    );
    return (res.leaders && res.leaders.length > 0) ? res.leaders : mock.mockBridgeKOLs;
  },

  async getCrossSegmentDiffusion(cascadeId = 'cascade_ai_breakthrough'): Promise<CrossSegmentDiffusionReport> {
    return safeFetch<CrossSegmentDiffusionReport>(
      `/api/v1/analytics/diffusion/cross-segment/${encodeURIComponent(cascadeId)}`,
      undefined,
      mock.mockDiffusionReport
    );
  },

  // Component F: Connectors
  async getConnectorsStatus(): Promise<Record<string, ConnectorInfo>> {
    const res = await safeFetch<{ connectors: Record<string, ConnectorInfo> }>(
      '/api/v1/connectors/status',
      undefined,
      { connectors: mock.mockConnectors }
    );
    return res.connectors || mock.mockConnectors;
  },

  async pollConnector(platform: string, query?: string, limit = 5, ingest = true): Promise<{ count: number; posts: CanonicalPost[] }> {
    try {
      const res = await fetch(`/api/v1/connectors/${platform}/poll?ingest=${ingest}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: query || null, limit }),
      });
      if (res.ok) {
        return await res.json();
      }
    } catch (e) {
      console.warn(`Poll for ${platform} fallback:`, e);
    }
    return { count: 2, posts: mock.mockPosts.slice(0, 2) };
  },
};
