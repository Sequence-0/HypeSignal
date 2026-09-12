# HypeSignal: System Architecture & Overall Working

**HypeSignal** is an offline-first, AI-driven Social Media Analytics Framework designed to process raw multi-platform social streams and benchmark corpora to produce audience insights across five core analytical pillars.

---

## 1. High-Level Architectural Blueprint

```
                     ┌─────────────────────────────────────────────────────────┐
                     │                 Data Sources & Feeds                    │
                     │  • Cheng-Caverlee-Lee (Geo)  • Lerman 2010 (Network)    │
                     │  • TweetEval Benchmarks       • Live Connector Stubs     │
                     └────────────────────────────┬────────────────────────────┘
                                                  │
                                                  ▼
                     ┌─────────────────────────────────────────────────────────┐
                     │          Ingestion & Schema Normalization               │
                     │  • GeoDatasetAdapter        • LermanDatasetAdapter      │
                     │  • CanonicalPost            • CanonicalUser             │
                     │  • CanonicalCascadeEvent    • CanonicalGraphEdge        │
                     └────────────────────────────┬────────────────────────────┘
                                                  │
                                                  ▼
                     ┌─────────────────────────────────────────────────────────┐
                     │                 Storage & Indexing Tier                 │
                     │  • DuckDB (Columnar OLAP: posts, users, cascades, edges)│
                     │  • Qdrant (HNSW Vector Store: Bio & Persona Embeddings) │
                     │  • Graph Store (NetworkX In-Memory / Memgraph Bolt)     │
                     └────────────────────────────┬────────────────────────────┘
                                                  │
         ┌──────────────────┬─────────────────────┼────────────────────┬──────────────────┐
         ▼                  ▼                     ▼                    ▼                  ▼
┌──────────────────┐┌──────────────────┐┌───────────────────┐┌──────────────────┐┌──────────────────┐
│   Component A    ││   Component B    ││    Component C    ││   Component D    ││   Component E    │
│    Timeline      ││   Sentiment &    ││   Demographics    ││ Trends & Topics  ││ Network Topology │
│   Management     ││  Emotion Engine  ││ Profiling Engine  ││     Engine       ││   & KOL Engine   │
│ • Slicing & Feed ││ • Cardiff RoBERTa││ • spaCy Geo NER   ││ • Tier-1 Z-Score ││ • PageRank & KOLs│
│ • Interval Buckts││ • Multi-Emotions ││ • Lingua LangID   ││   Burst Radar    ││ • Louvain Clusters│
│ • Replay Streams ││ • Irony Inversion││ • MiniLM Personas ││ • Tier-2 BERTopic││ • Diffusion Tree │
│ • Cascade Traces ││ • Stance Tracking││ • Circadian Tiers ││ • Trend Ranking  ││ • Subgraph Export│
└────────┬─────────┘└────────┬─────────┘└─────────┬─────────┘└────────┬─────────┘└────────┬─────────┘
         │                   │                    │                   │                   │
         └───────────────────┴────────────────────┼───────────────────┴───────────────────┘
                                                  │
                                                  ▼
                     ┌─────────────────────────────────────────────────────────┐
                     │          Unified FastAPI REST & Streaming API           │
                     │  • /api/v1/timeline       • /api/v1/sentiment           │
                     │  • /api/v1/demographics   • /api/v1/trends              │
                     │  • /api/v1/network        • /api/v1/connectors          │
                     │  • /health & /docs (Interactive OpenAPI / Swagger)      │
                     └─────────────────────────────────────────────────────────┘
```

---

## 2. Core Components & Working

### Component A: Continuous Data Collection & Timeline Manager
- **Storage Layer**: Embedded columnar DuckDB instance storing `posts`, `users`, `cascade_events`, and `graph_edges`.
- **Query Capabilities**:
  - Arbitrary sliding/tumbling time-window slicing: `get_timeline_slice(start_time, end_time, platform, limit)`.
  - Chronological replay iterators simulating streaming data from historical records.
  - Granular time-bucketed activity volume aggregations using DuckDB `time_bucket(INTERVAL '...', timestamp)`.
  - Chronological cascade propagation sequences with relative lag times in seconds.

### Component B: Multi-Dimensional Sentiment & Emotion Inference Engine
- **Models**:
  - **Sentiment**: Cardiff NLP Twitter-RoBERTa base model (`cardiffnlp/twitter-roberta-base-sentiment-latest`) detecting positive, neutral, and negative sentiment.
  - **Emotions**: DistilRoBERTa (`j-hartmann/emotion-english-distilroberta-base`) classifying 7 nuanced emotions: joy, optimism, anger, sadness, fear, surprise, and disgust.
  - **Irony / Sarcasm**: Cardiff NLP irony detector (`cardiffnlp/twitter-roberta-base-irony`). When irony exceeds the confidence threshold (0.85), positive sentiment polarity is inverted to negative.
  - **Stance**: Cardiff NLP target-specific stance models (`climate`, `abortion`, `atheism`, `feminist`, `hillary`).
- **Temporal Tracking**: `TemporalSentimentTracker` enriches historical DuckDB posts across time buckets to compute rolling sentiment polarity, sarcasm index, and emotional trajectories over time.

### Component C: Automated Demographic Profiling Engine
- **Geographic Profiling**:
  - spaCy NER (`en_core_web_sm`) extracts GPE/LOC entities from user profile locations and tweet texts.
  - Normalizes raw text against gazetteers (major world cities, US states) and parses raw GPS coordinates (`UT: lat, lon`).
- **Linguistic Profiling**: Fast language identification to assess audience language distributions.
- **Persona & Professional Interest Profiling**:
  - Zero-shot semantic embeddings generated via `sentence-transformers/all-MiniLM-L6-v2`.
  - Maps user bios and sample post texts against a standard persona taxonomy (Tech & Engineering, Finance & Crypto, Media & Journalism, Academia & Research, Healthcare, Creative Arts, etc.).
  - Vector indexing in Qdrant (in-memory or on-disk).
- **Behavioral Profiling**: Calculates posting velocity, active hours (circadian distribution), and engagement tiers (`casual`, `active`, `power_user`).

### Component D: Real-Time Trends & Dynamic Topic Detection Engine
- **Tier-1 Statistical Burst Detector**:
  - Computes term frequency acceleration and Poisson/Z-score deviation over historical moving baseline windows in DuckDB.
  - Triggers instant alerts when word/hashtag velocity deviates significantly (Z-score > threshold).
- **Tier-2 Dynamic Temporal Topic Modeler**:
  - Employs BERTopic with `all-MiniLM-L6-v2` embeddings and c-TF-IDF representations.
  - Automatically models dynamic topic evolution over time (`topics_over_time`).
  - Includes KMeans/CountVectorizer fallbacks for small/medium corpora to prevent silent dropping of small clusters.
- **Global Trend Ranker**:
  - Composite scoring integrating burst acceleration (50%), sentiment intensity (25%), and participant diversity (25%).

### Component E: Link Analysis & Network Topology Engine
- **Pluggable Graph Storage**:
  - `NetworkXGraphStore`: High-performance in-memory directed graph with edge weights and revision tracking.
  - `MemgraphStore`: Bolt-protocol adapter for production graph databases (Memgraph / Neo4j) with parameterized queries and strict relationship type validation against Cypher injection.
- **Key Opinion Leader (KOL) Ranking**:
  - Composite multi-factor ranking combining PageRank, follower in-degree, and betweenness centrality.
  - Validated against Lerman degree distribution ground truth via Spearman rank correlation.
- **Community Detection**:
  - Louvain modularity optimization and Label Propagation Algorithm (LPA).
- **Information Diffusion Cascade Tracer**:
  - Reconstructs diffusion trees from timestamped adoption events and follower graph topology.
  - Computes structural virality (Wiener index), propagation depth/breadth, half-life, and velocity.
- **Caching & Consistency**:
  - Graph revision-aware caching ensures hot endpoints (`/kols`, `/overview`, `/user/{id}`) return instantly and update automatically when edges are added.

### Live Platform Ingestion Connectors (`src/hypesignal/connectors/`)
- **Base Architecture**: Abstract `PlatformConnector` with thread-safe lifecycle states (`connect`, `disconnect`), token bucket rate limiting (`TokenBucketRateLimiter`), fallback entity extraction (hashtags, mentions, URLs), and atomic stats recording.
- **Platform Stubs**:
  - `TwitterConnector`: Normalizes Twitter v2 API Tweet objects.
  - `RedditConnector`: Normalizes Reddit submission and comment JSON payloads.
  - `YouTubeConnector`: Normalizes YouTube Data API v3 `CommentThread` items.
  - `TelegramConnector`: Normalizes Telegram MTProto / Bot API message objects.

---

## 3. Unified REST API Endpoints

The API is served via FastAPI with automatic OpenAPI docs (`/docs` and `/redoc`) and CORS security:

| Route | Method | Description |
|---|---|---|
| `/health`, `/api/v1/health` | `GET` | Health check returning status of all 5 engines and storage |
| `/api/v1/timeline/bounds` | `GET` | Earliest, latest, and total post bounds |
| `/api/v1/timeline/slice` | `GET` | Chronological slice of posts within a time window |
| `/api/v1/timeline/timeseries` | `GET` | Activity post volume aggregated into time buckets |
| `/api/v1/timeline/cascade/{id}` | `GET` | Chronological adoption trace of a diffusion cascade |
| `/api/v1/timeline/user/{id}` | `GET` | Chronological post history for an individual user |
| `/api/v1/sentiment/analyze` | `POST` | Multi-dimensional NLP inference on text snippets (max batch 100) |
| `/api/v1/sentiment/temporal` | `GET` | Chronological sentiment/emotion trajectory from DuckDB |
| `/api/v1/demographics/profile-user` | `POST` | Profile a single user (geo, persona, behavior) |
| `/api/v1/demographics/breakdown` | `GET` | Population-level breakdown (countries, cities, languages, personas) |
| `/api/v1/demographics/users` | `GET` | List profiled users from DuckDB |
| `/api/v1/trends/overview` | `GET` | Dual-tier trend overview (active bursts + dynamic topics + ranking) |
| `/api/v1/trends/bursts` | `GET` | Tier-1 statistical burstiness alerts |
| `/api/v1/trends/topics` | `GET` | Tier-2 BERTopic topic representations and timelines |
| `/api/v1/network/overview` | `GET` | Complete network overview (density, top KOLs, communities, cascades) |
| `/api/v1/network/kols` | `GET` | Influencer leaderboard with community clusters and centrality |
| `/api/v1/network/user/{id}` | `GET` | Individual user's network profile and influence rank |
| `/api/v1/network/subgraph/{id}` | `GET` | Egocentric neighborhood subgraph export for visualizers |
| `/api/v1/network/cascade/{id}` | `GET` | Reconstructed diffusion cascade tree and structural virality |
| `/api/v1/connectors/status` | `GET` | Operational status and rate limiter metrics for all connectors |
| `/api/v1/connectors/{platform}/poll` | `POST` | Trigger live connector polling with optional DB ingestion |
