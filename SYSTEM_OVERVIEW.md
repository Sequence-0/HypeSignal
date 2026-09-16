# HypeSignal: System Architecture & Overall Working

**HypeSignal** is an offline-first, AI-driven Social Media Analytics Framework designed to process raw multi-platform social streams and benchmark corpora to produce audience insights across five core analytical pillars and real-time streaming pipelines.

---

## 1. High-Level Architectural Blueprint

```
                     ┌──────────────────────────────────────────────────────────────────┐
                     │                      Data Sources & Feeds                        │
                     │  • Cheng-Caverlee-Lee (Geo)        • Lerman 2010 (Network)       │
                     │  • TweetEval Benchmarks            • Bluesky AT Protocol (Live)  │
                     │  • YouTube Data API v3 (Live)      • Telegram MTProto (Live)     │
                     │  • Twitter & Reddit Connectors     • Mock Synthetic Generators   │
                     └────────────────────────────────┬─────────────────────────────────┘
                                                      │
                                                      ▼
                     ┌──────────────────────────────────────────────────────────────────┐
                     │               Ingestion & Schema Normalization                   │
                     │  • GeoDatasetAdapter             • LermanDatasetAdapter          │
                     │  • CanonicalPost                 • CanonicalUser                 │
                     │  • CanonicalCascadeEvent         • CanonicalGraphEdge            │
                     │  • ConversationThreadManager (Parent-Child Hierarchy Reconstruct)│
                     └────────────────────────────────┬─────────────────────────────────┘
                                                      │
                                                      ▼
                     ┌──────────────────────────────────────────────────────────────────┐
                     │                    Storage & Indexing Tier                       │
                     │  • DuckDB (Columnar OLAP: posts, users, analytics, cascades)     │
                     │  • Qdrant (HNSW Vector Store: Bio & Persona Embeddings)          │
                     │  • Graph Store (NetworkX In-Memory / Memgraph Bolt Cypher)       │
                     └────────────────────────────────┬─────────────────────────────────┘
                                                      │
          ┌──────────────────┬────────────────────────┼───────────────────────┬──────────────────┐
          ▼                  ▼                        ▼                       ▼                  ▼
┌──────────────────┐┌───────────────────┐┌────────────────────────┐┌───────────────────┐┌──────────────────┐
│   Component A    ││    Component B    ││      Component C       ││    Component D    ││   Component E    │
│    Timeline      ││    Sentiment &    ││      Demographics      ││  Trends, Topics   ││ Network Topology │
│   Management     ││  Emotion Engine   ││    Profiling Engine    ││   & Kinematics    ││   & KOL Engine   │
│ • Slicing & Feed ││ • Cardiff RoBERTa ││ • spaCy Geo NER        ││ • Tier-1 Z-Score  ││ • PageRank & KOLs│
│ • Interval Buckts││ • Nuanced Emotions││ • Lingua LangID        ││   Burst Radar     ││ • Louvain & LPA  │
│ • Replay Streams ││ • Irony Inversion ││ • MiniLM Personas      ││ • Tier-2 BERTopic ││ • Cascade Tracer │
│ • Discussion Tree││ • Target Stance   ││ • Multi-Stage Age      ││ • Kinematics v, a ││ • Bridge KOLs    │
│   Reconstruction ││ • Thread Dynamics ││ • Influencer Profiler  ││ • Narrative Drift ││ • Cross-Segment  │
│   (Depth/Latency)││   (Drift/Controv) ││   (k-Anonymity)        ││ • Trend Ranking   ││   Diffusion      │
└────────┬─────────┘└─────────┬─────────┘└───────────┬────────────┘└─────────┬─────────┘└────────┬─────────┘
         │                    │                       │                       │                   │
         └────────────────────┴───────────────────────┼───────────────────────┴───────────────────┘
                                                      │
                                                      ▼
                     ┌──────────────────────────────────────────────────────────────────┐
                     │            Orchestration & Streaming Pipeline Tier               │
                     │  • AnalyticsPipelineOrchestrator (Background Async Tick Loop)    │
                     │  • Concurrent Ingestion (asyncio.gather across enabled feeds)    │
                     │  • Non-Blocking Inference (asyncio.to_thread PyTorch offload)    │
                     │  • EventBroadcaster (Server-Sent Events with Bounded Backpressure│
                     └────────────────────────────────┬─────────────────────────────────┘
                                                      │
                                                      ▼
                     ┌──────────────────────────────────────────────────────────────────┐
                     │              Unified FastAPI REST & Streaming API                │
                     │  • /api/v1/timeline           • /api/v1/sentiment                │
                     │  • /api/v1/demographics       • /api/v1/trends                   │
                     │  • /api/v1/network            • /api/v1/connectors               │
                     │  • /api/v1/analytics/* (Consolidated Phase 2 Deep Analytics)     │
                     │  • /api/v1/streaming/events (Real-Time Server-Sent Events)       │
                     │  • /health & /docs (Interactive OpenAPI / Swagger)               │
                     └──────────────────────────────────────────────────────────────────┘
```

---

## 2. Core Components & Working

### Component A: Continuous Data Collection & Timeline Manager
- **Storage Layer**: Embedded columnar DuckDB instance storing relational analytical tables: `posts`, `users`, `post_analytics`, `user_communities`, `cascade_events`, and `graph_edges`.
- **Query Capabilities**:
  - Arbitrary sliding/tumbling time-window slicing: `get_timeline_slice(start_time, end_time, platform, limit)`.
  - Chronological replay iterators simulating streaming data from historical records.
  - Granular time-bucketed activity volume aggregations using DuckDB `time_bucket(INTERVAL '...', timestamp)`.
  - Chronological cascade propagation sequences with relative lag times in seconds.
- **Discussion Tree Reconstruction (`ConversationThreadManager`)**:
  - Reconstructs full nested conversation trees from raw parent-child reply pointers (`parent_id`).
  - Implements cycle-safe Breadth-First Search (BFS) traversal.
  - Computes structural conversational metrics: max tree depth, breadth distribution, average reply latency, branching factor, and participant diversity.

### Component B: Multi-Dimensional Sentiment & Emotion Inference Engine
- **Inference Pipeline**:
  - **Sentiment Polarity**: Cardiff NLP Twitter-RoBERTa base model (`cardiffnlp/twitter-roberta-base-sentiment-latest`) detecting positive, neutral, and negative sentiment.
  - **Calibrated Nuanced Emotions**: DistilRoBERTa (`j-hartmann/emotion-english-distilroberta-base`) producing 10 emotions: joy, optimism, anger, sadness, fear, surprise, disgust, anxiety, and excitement.
  - **Irony / Sarcasm Inversion**: Cardiff NLP irony detector (`cardiffnlp/twitter-roberta-base-irony`). When irony exceeds the confidence threshold (0.85), positive surface sentiment is inverted to negative, adjusting the composite score.
  - **Zero-Shot Stance**: Cardiff NLP target-specific stance models (`climate`, `abortion`, `atheism`, `feminist`, `hillary`), with semantic polarity fallback for arbitrary targets.
- **Temporal Sentiment Tracking**: `TemporalSentimentTracker` enriches historical DuckDB posts across time buckets using reused connection caching and thread-safe Arrow table registration, computing rolling sentiment polarity, sarcasm index, and emotional trajectories.
- **Thread Dynamics Analyzer (`ThreadSentimentAnalyzer`)**:
  - Analyzes sentiment and emotional propagation along discussion trees.
  - Computes root-to-leaf polarity drift, controversy index (bimodal polarity variance), hostility velocity, and dominant trajectory states.

### Component C: Automated Demographic Profiling Engine
- **Geographic Profiling**:
  - spaCy NER (`en_core_web_sm`) extracts GPE/LOC entities from user profile locations and tweet texts.
  - Normalizes raw text against gazetteers (major world cities, US states) and parses raw GPS coordinates (`UT: lat, lon`).
- **Linguistic Profiling**: Fast language identification to assess audience language distributions.
- **Persona & Professional Interest Profiling**:
  - Zero-shot semantic embeddings generated via `sentence-transformers/all-MiniLM-L6-v2`.
  - Maps user bios and sample post texts against standard persona taxonomies (Tech & AI, Finance & Crypto, Media & Journalism, Healthcare, Creative Arts, etc.).
  - Vector indexing in Qdrant (in-memory or on-disk).
- **Multi-Stage Age Classifier (`MultiStageAgeClassifier`)**:
  - **Stage 1**: Explicit regex markers for ages, birth years, and life-stage keywords with 0.90 confidence short-circuit.
  - **Stage 2**: Bio vector semantic anchor cosine similarity across age bracket prototypes.
  - **Stage 3**: Zero-shot NLI post text scoring.
  - **Stage 4**: Weighted ensembling mapping users into standard brackets: `<18`, `18-24`, `25-34`, `35-49`, `50-64`, and `65+`.
- **Influencer Audience Profiler (`InfluencerAudienceProfiler`)**:
  - Aggregates demographics across an influencer's follower graph (age distributions, top personas, geo concentrations, and active tiers).
  - Enforces $k$-anonymity privacy filters (`min_group_size`) to protect individual follower privacy.

### Component D: Real-Time Trends & Dynamic Topic Detection Engine
- **Tier-1 Statistical Burst Detector**:
  - Computes term frequency acceleration and Poisson/Z-score deviation over historical moving baseline windows in DuckDB.
  - Triggers instant alerts when word/hashtag velocity deviates significantly (Z-score > threshold).
- **Tier-2 Dynamic Temporal Topic Modeler**:
  - Employs BERTopic with `all-MiniLM-L6-v2` embeddings and c-TF-IDF representations.
  - Automatically models dynamic topic evolution over time (`topics_over_time`).
  - Includes KMeans/CountVectorizer fallbacks for small/medium corpora to prevent silent dropping of small clusters.
- **Predictive Trend Kinematics Forecaster (`TrendForecaster`)**:
  - 1st derivative (velocity) and 2nd derivative (acceleration) post volume tracking across temporal sliding windows.
  - Classifies trend trajectories into 6 lifecycle states: `EMERGING`, `VIRAL_SURGE`, `PEAKING`, `DECELERATING`, `STABLE`, and `DORMANT`.
  - Computes composite virality potential score (velocity, acceleration, and unique author diversity).
- **Narrative Drift Tracker (`NarrativeDriftTracker`)**:
  - Tracks semantic drift and sentiment inversions between baseline and current conversational windows.
  - Computes cosine distance between normalized sentence embedding centroids and signed sentiment delta.
- **Global Trend Ranker**:
  - Composite scoring integrating burst acceleration (50%), sentiment intensity (25%), and participant diversity (25%).

### Component E: Link Analysis & Network Topology Engine
- **Pluggable Graph Storage**:
  - `NetworkXGraphStore`: High-performance in-memory directed graph with edge weights, revision tracking, and subgraph extraction.
  - `MemgraphStore`: Bolt-protocol adapter for production graph databases (Memgraph / Neo4j) with parameterized queries and strict relationship type validation against Cypher injection.
- **Key Opinion Leader (KOL) Ranking**:
  - Composite multi-factor ranking combining PageRank, follower in-degree, and betweenness centrality.
  - Validated against Lerman degree distribution ground truth via Spearman rank correlation.
- **Boundary-Spanning Bridge KOLs (`BridgeKOLAnalyzer`)**:
  - Computes betweenness centrality (with 500-pivot sampling for scaling), cross-community edge ratio, and inter-cluster reach to detect boundary-spanning influencers.
- **Community Detection**:
  - Louvain modularity optimization and Label Propagation Algorithm (LPA).
- **Cascade Diffusion & Cross-Segment Tracking (`CrossSegmentDiffusionTracker`)**:
  - Reconstructs diffusion trees from timestamped adoption events and follower graph topology.
  - Computes structural virality (Wiener index), propagation depth/breadth, half-life, and velocity.
  - Traces adoption chronology across demographic age groups and Louvain graph communities.
- **Caching & Consistency**:
  - Graph revision-aware caching ensures hot endpoints (`/kols`, `/overview`, `/bridge-kols`) return instantly and update automatically when edges are mutated.

### Component F: Pipeline Orchestration & Real-Time Streaming Tier
- **Analytics Pipeline Orchestrator (`AnalyticsPipelineOrchestrator`)**:
  - Asynchronous background tick loop coordinating continuous data collection and analytics.
  - Concurrent ingestion: executes `connector.poll()` concurrently across all enabled platform connectors using `asyncio.gather`.
  - Non-blocking PyTorch offload: dispatches heavy transformer inference via `asyncio.to_thread` to keep the event loop responsive.
  - Automated alerting: evaluates burst anomalies and broadcasts real-time alerts.
- **Server-Sent Events (SSE) Broadcaster (`EventBroadcaster`)**:
  - High-throughput async event broadcast hub at `/api/v1/streaming/events`.
  - Implements bounded client queues (`maxsize=100`) with drop-oldest backpressure strategy.
  - Handles client disconnects cleanly and sends periodic keepalive pings.

### Live Platform Ingestion Connectors (`src/hypesignal/connectors/`)
- **Base Architecture**: Abstract `PlatformConnector` with thread-safe lifecycle states (`connect`, `disconnect`), token bucket rate limiting (`TokenBucketRateLimiter`), fallback entity extraction, and atomic metrics recording.
- **Implemented Connectors**:
  - `BlueskyConnector`: Native HTTPX connector querying public Bluesky AT Protocol endpoints (`searchPosts`, `getPostThread`) with bounded recursion depth. Zero API keys required.
  - `YouTubeConnector`: Live comment thread and nested reply ingestion via YouTube Data API v3 with daily quota tracking (5,000 unit budget ceiling) and Template Method `_do_disconnect` socket teardown.
  - `TelegramConnector`: Native Telethon MTProto client running on a dedicated background event loop thread (`_TelegramLoopThread`), supporting StringSessions, bot tokens, test DC servers, double-safe timestamp parsing, and deadlock-free GC teardown.
  - `TwitterConnector`: Normalizes Twitter v2 API Tweet objects with fallback synthetic generator.
  - `RedditConnector`: Normalizes Reddit submission and comment JSON payloads with fallback synthetic generator.
- **Configuration & Secret Shielding (`src/hypesignal/config.py`)**:
  - Zero-dependency `.env` file parser supporting inline comment stripping, variable exports, and quote preservation.
  - Credentials, tokens, and `*.session` files shielded via `.gitignore`.

---

## 3. Unified REST & Streaming API Endpoints

The API is served via FastAPI with automatic OpenAPI docs (`/docs` and `/redoc`), explicit CORS security, and sanitized 500 error handling:

| Route | Method | Pillar | Description |
|---|---|:---:|---|
| `/health`, `/api/v1/health` | `GET` | Core | Health check returning status of all engines and storage |
| `/api/v1/timeline/bounds` | `GET` | A | Earliest, latest, and total post bounds in DuckDB |
| `/api/v1/timeline/slice` | `GET` | A | Chronological slice of posts within a time window |
| `/api/v1/timeline/timeseries` | `GET` | A | Activity post volume aggregated into time buckets |
| `/api/v1/timeline/cascade/{id}` | `GET` | A | Chronological adoption trace of a diffusion cascade |
| `/api/v1/timeline/user/{id}` | `GET` | A | Chronological post history for an individual user |
| `/api/v1/sentiment/analyze` | `POST` | B | Multi-dimensional NLP inference on text snippets (max batch 100) |
| `/api/v1/sentiment/temporal` | `GET` | B | Chronological sentiment/emotion trajectory from DuckDB |
| `/api/v1/demographics/profile-user` | `POST` | C | Profile a single user (geo, persona, behavior) |
| `/api/v1/demographics/breakdown` | `GET` | C | Population-level breakdown (countries, cities, languages, personas) |
| `/api/v1/demographics/users` | `GET` | C | List profiled users from DuckDB |
| `/api/v1/trends/overview` | `GET` | D | Dual-tier trend overview (active bursts + dynamic topics + ranking) |
| `/api/v1/trends/bursts` | `GET` | D | Tier-1 statistical burstiness alerts |
| `/api/v1/trends/topics` | `GET` | D | Tier-2 BERTopic topic representations and timelines |
| `/api/v1/network/overview` | `GET` | E | Complete network overview (density, top KOLs, communities, cascades) |
| `/api/v1/network/kols` | `GET` | E | Influencer leaderboard with community clusters and centrality |
| `/api/v1/network/user/{id}` | `GET` | E | Individual user's network profile and influence rank |
| `/api/v1/network/subgraph/{id}` | `GET` | E | Egocentric neighborhood subgraph export for visualizers |
| `/api/v1/network/cascade/{id}` | `GET` | E | Reconstructed diffusion cascade tree and structural virality |
| `/api/v1/analytics/conversation/{id}` | `GET` | A & B | Discussion tree hierarchy, reply latency, branching, and polarity drift |
| `/api/v1/analytics/influencer/{id}` | `GET` | C & E | Influencer follower demographics, age distribution, and $k$-anonymity |
| `/api/v1/analytics/trends/forecast` | `POST` | D | 1st/2nd derivative kinematics (velocity, acceleration, 6-state lifecycle) |
| `/api/v1/analytics/trends/drift` | `POST` | D | Narrative semantic drift and signed sentiment inversions |
| `/api/v1/analytics/diffusion/{id}` | `GET` | E | Cross-segment diffusion order across age groups and graph communities |
| `/api/v1/analytics/network/bridge-kols` | `GET` | E | Boundary-spanning Bridge KOL leaderboard with cross-community ratios |
| `/api/v1/connectors/status` | `GET` | Live | Operational status and rate limiter metrics for all connectors |
| `/api/v1/connectors/{platform}/poll` | `POST` | Live | Trigger live connector polling with optional DB ingestion |
| `/api/v1/streaming/events` | `GET` | SSE | Real-time Server-Sent Events stream with bounded backpressure |
