/**
 * Canonical types and schemas for HypeSignal Frontend
 * Matches FastAPI backend models and problem statement requirements
 */

export type Platform = 'twitter' | 'telegram' | 'youtube' | 'reddit' | 'bluesky';

export interface PostMetrics {
  likes: number;
  reposts: number;
  replies: number;
  quotes: number;
  impressions?: number;
  views?: number;
}

export interface CanonicalPost {
  id: string;
  platform: Platform | string;
  author_id: string;
  author_screen_name?: string;
  text: string;
  timestamp: string;
  parent_id?: string;
  reply_to_user_id?: string;
  source_client?: string;
  urls: string[];
  hashtags: string[];
  mentions: string[];
  media_urls: string[];
  metrics: PostMetrics;
  extra_metadata?: Record<string, any>;
}

export interface TimelineBounds {
  earliest?: string;
  latest?: string;
  total_posts: number;
}

export interface ActivityTimeseriesPoint {
  bucket: string;
  post_count: number;
}

export interface ActivityTimeseriesResponse {
  interval: string;
  total_buckets: number;
  points: ActivityTimeseriesPoint[];
}

export interface NuancedEmotions {
  joy: number;
  optimism: number;
  anger: number;
  sadness: number;
  fear: number;
  surprise: number;
  disgust: number;
  anxiety: number;
  excitement: number;
  [key: string]: number;
}

export interface SentimentAnalyzeResult {
  text: string;
  sentiment: 'positive' | 'neutral' | 'negative';
  sentiment_score: number;
  emotions: NuancedEmotions;
  primary_emotion: string;
  irony_score: number;
  is_ironic: boolean;
  effective_polarity: 'positive' | 'neutral' | 'negative';
  irony_inverted: boolean;
  stance_target?: string;
  stance?: 'supportive' | 'against' | 'neutral' | 'none';
  stance_score?: number;
}

export interface TemporalSentimentPoint {
  bucket: string;
  post_count: number;
  mean_sentiment_score?: number;
  positive_ratio?: number;
  negative_ratio?: number;
  neutral_ratio?: number;
  sarcasm_rate?: number;
  mean_irony_score?: number;
  dominant_emotion?: string;
}

export interface DemographicDistributionItem {
  country?: string;
  city?: string;
  language?: string;
  persona?: string;
  count: number;
  percentage: number;
}

export interface AggregateDemographics {
  total_users_profiled: number;
  top_countries: Array<{ country: string; count: number; percentage: number }>;
  top_cities: Array<{ city: string; count: number; percentage: number }>;
  top_languages: Array<{ language: string; count: number; percentage: number }>;
  persona_distribution: Record<string, number>;
  age_distribution: Record<string, number>;
}

export interface UserProfile {
  user_id: string;
  screen_name: string;
  inferred_age_bracket?: string;
  age_confidence?: number;
  detected_locations: string[];
  normalized_country?: string;
  normalized_city?: string;
  detected_languages: string[];
  primary_persona?: string;
  persona_scores: Record<string, number>;
  top_interests: string[];
  post_count: number;
}

export interface InfluencerAudienceProfile {
  influencer_id: string;
  total_followers_analyzed: number;
  min_group_size: number;
  privacy_filter_active: boolean;
  age_distribution: Record<string, number>;
  top_personas: Array<{ persona: string; count: number; percentage: number }>;
  geo_concentration: Array<{ location: string; count: number; percentage: number }>;
  activity_tiers: Record<string, number>;
}

export type TrendLifecycleState = 'EMERGING' | 'VIRAL_SURGE' | 'PEAKING' | 'DECELERATING' | 'STABLE' | 'DORMANT';

export interface BurstAlert {
  term: string;
  current_count: number;
  baseline_mean: number;
  baseline_std: number;
  z_score: number;
  is_burst: boolean;
  window_end: string;
}

export interface DynamicTopic {
  topic_id: number;
  name: string;
  top_words: Array<{ word: string; weight: number }>;
  doc_count: number;
}

export interface TrendForecast {
  term: string;
  velocity: number;
  acceleration: number;
  lifecycle_state: TrendLifecycleState;
  virality_potential_score: number;
  author_diversity_ratio: number;
  recent_volumes: number[];
}

export interface NarrativeDriftAlert {
  term: string;
  drift_detected: boolean;
  cosine_distance: number;
  sentiment_inverted: boolean;
  baseline_sentiment: number;
  current_sentiment: number;
  drift_magnitude: number;
}

export interface KOLProfile {
  user_id: string;
  screen_name?: string;
  composite_influence_score: number;
  pagerank: number;
  in_degree: number;
  out_degree: number;
  betweenness_centrality: number;
  community_id: number;
}

export interface BridgeKOL {
  user_id: string;
  screen_name?: string;
  betweenness_centrality: number;
  cross_community_ratio: number;
  home_community: number;
  connected_communities: number[];
  total_neighbors: number;
}

export interface NetworkOverview {
  total_nodes: number;
  total_edges: number;
  density: number;
  community_count: number;
  top_kols: KOLProfile[];
}

export interface TreeNode {
  post_id: string;
  author_id: string;
  text: string;
  timestamp: string;
  depth: number;
  replies: TreeNode[];
  sentiment?: 'positive' | 'neutral' | 'negative';
}

export interface ConversationThread {
  conversation_id: string;
  metrics: {
    total_posts: number;
    total_replies: number;
    max_depth: number;
    avg_branching_factor: number;
    participant_count: number;
    participant_ids: string[];
    duration_seconds: number;
    mean_reply_latency_seconds: number;
    depth_distribution: Record<string, number>;
  };
  sentiment_dynamics: {
    root_post_id?: string;
    root_polarity?: string;
    root_sentiment_score?: number;
    root_primary_emotion?: string;
    comment_mean_sentiment?: number;
    polarity_drift?: number;
    controversy_index?: number;
    hostility_velocity?: number;
    supportive_ratio?: number;
    against_ratio?: number;
    neutral_ratio?: number;
    dominant_thread_emotion?: string;
    emotional_trajectory?: Array<{ depth: number; primary_emotion: string; polarity_score: number }>;
  };
  thread_tree: TreeNode;
}

export interface CascadeDiffusionEvent {
  segment: string;
  user_id: string;
  seconds_since_origin: number;
  sentiment_score?: number;
  effective_polarity?: string;
}

export interface CrossSegmentDiffusionReport {
  cascade_id: string;
  segment_by: string;
  total_events: number;
  structural_virality_wiener: number;
  propagation_depth: number;
  half_life_seconds: number;
  adoption_timeline: CascadeDiffusionEvent[];
}

export interface ConnectorInfo {
  platform: string;
  enabled: boolean;
  connected: boolean;
  requests_recorded: number;
  max_requests_per_minute: number;
}

export interface HealthResponse {
  status: string;
  version: string;
  engines: Record<string, string>;
}

export interface LiveStreamEvent {
  id: string;
  type: 'post' | 'burst' | 'alert' | 'system';
  platform?: string;
  title: string;
  detail: string;
  timestamp: string;
}
