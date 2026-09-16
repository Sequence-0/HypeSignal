# HypeSignal

**HypeSignal** is an offline-first, AI-driven Social Media Analytics Framework. It processes multi-platform social feeds and benchmark corpora to deliver audience insights across five core analytical pillars and real-time streaming pipelines:

1. **Continuous Timeline & Conversation Thread Management**: Unified canonical schema and chronological query engine backed by DuckDB, featuring thread hierarchy reconstruction (depth, branching factor, reply latency).
2. **Multi-Dimensional Sentiment & Thread Dynamics**: Zero-shot Cardiff RoBERTa sentiment polarity, calibrated nuanced emotion detection (joy, optimism, anger, sadness, fear, surprise, disgust, anxiety, excitement), sarcasm score inversion, target stance classification, and conversation polarity drift tracking.
3. **Automated Demographic & Influencer Profiling**: Content-based geolocation extraction (spaCy NER), bio-persona interest clustering (Sentence-Transformers + Qdrant), multi-stage age classification (18-24, 25-34, 35-49, 50+), and $k$-anonymity influencer audience profiling.
4. **Predictive Trend Kinematics & Dynamic Topics**: 1st/2nd derivative velocity and acceleration tracking across sliding windows, 6 lifecycle states (Emerging, Viral Surge, Peaking, Decelerating, Stable, Dormant), narrative semantic drift tracking, statistical Z-score burst detection, and temporal topic modeling (BERTopic).
5. **Link Analysis, Bridge KOLs & Cascade Diffusion**: Follower graph topology, Key Opinion Leader (KOL) ranking, community detection (Louvain/LPA), boundary-spanning Bridge KOL identification, and cross-segment cascade diffusion tracing across demographic brackets and graph clusters.
6. **Asynchronous Orchestration & Real-Time SSE**: Background tick-based pipeline orchestrator coordinating concurrent connector ingestion, non-blocking NLP inference, and Server-Sent Events (SSE) streaming with bounded backpressure.

For an in-depth architectural breakdown of every engine, refer to [SYSTEM_OVERVIEW.md](SYSTEM_OVERVIEW.md).

---

## Quick Start

### 1. Prerequisites & Environment Setup

HypeSignal uses **Python 3.14** and is managed via **UV**:

```bash
# Install dependencies into virtual environment
uv sync
```

### 2. Environment Configuration & Secret Management

Copy the environment template and configure optional API keys:

```bash
cp .env.example .env
```

HypeSignal runs out-of-the-box in **offline synthetic mock mode** with zero API keys required. You can optionally supply credentials for live data acquisition:
- **Bluesky**: Public AT Protocol endpoints require zero credentials.
- **YouTube**: Set `YOUTUBE_API_KEY` for live Data API v3 comments and search.
- **Telegram**: Set `TELEGRAM_API_ID` and `TELEGRAM_API_HASH` (or `TELEGRAM_BOT_TOKEN`) for native MTProto ingestion.

### 3. Ingesting Data into Persistent DuckDB

HypeSignal comes with an automated ingestion script [`scripts/ingest_offline_datasets.py`](scripts/ingest_offline_datasets.py) that loads offline corpora (`Cheng-Caverlee-Lee Geo`, `Lerman 2010 Twitter cascades & follower edges`) into a persistent DuckDB database file:

```bash
# Ingest sample records (5,000 posts, 5,000 users, 5,000 cascades, 10,000 follower edges)
uv run python scripts/ingest_offline_datasets.py --db-path data/hypesignal.duckdb

# Or ingest full datasets without row limits:
uv run python scripts/ingest_offline_datasets.py --db-path data/hypesignal.duckdb --full
```

The script will report progress and output a summary of total records stored:
```text
DuckDB Database Summary for 'data/hypesignal.duckdb':
  Total Posts:          5000
  Total Users:          7250
  Total Cascade Events: 5000
  Total Graph Edges:    10000
```

---

## Running the Application

### 1. Launching the API Server

#### With Persistent DuckDB Data
Point the server to your ingested database file:
```bash
HYPESIGNAL_DB_PATH="data/hypesignal.duckdb" uv run hypesignal
```
Or run directly via Uvicorn:
```bash
HYPESIGNAL_DB_PATH="data/hypesignal.duckdb" uv run uvicorn hypesignal.api:app --host 0.0.0.0 --port 8000 --reload
```

#### In-Memory Mode (Stateless / Fresh Start)
If no database path is specified, HypeSignal boots with an empty in-memory store (`:memory:`):
```bash
uv run hypesignal
```

---

## Interactive API Documentation

Once the server is running, navigate to:
- **Swagger UI**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **ReDoc**: [http://localhost:8000/redoc](http://localhost:8000/redoc)
- **Health Check**: [http://localhost:8000/health](http://localhost:8000/health)

---

## Example API Queries

### Health Check
```bash
curl -s http://localhost:8000/health | jq
```

### Multi-Dimensional Sentiment Analysis
Zero-shot sentiment polarity, calibrated emotion probabilities, sarcasm inversion, and stance:
```bash
curl -s -X POST http://localhost:8000/api/v1/sentiment/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "text": "I am thrilled and ecstatic about the breakthrough research in foundation models! #AI"
  }' | jq
```

### Conversation Thread Analytics & Sentiment Dynamics
Reconstruct discussion trees, participant diversity, polarity drift, and controversy index:
```bash
curl -s "http://localhost:8000/api/v1/analytics/conversation/p_0" | jq
```

### Influencer Audience Demographics & Age Profiling
Retrieve follower age distributions, top personas, and active tiers with $k$-anonymity privacy filters:
```bash
curl -s "http://localhost:8000/api/v1/analytics/influencer/user_alice?min_group_size=5" | jq
```

### Predictive Trend Kinematics (Velocity & Acceleration)
Evaluate 1st/2nd derivative kinematics, 6-state lifecycle trajectory, and virality potential score:
```bash
curl -s -X POST http://localhost:8000/api/v1/analytics/trends/forecast \
  -H "Content-Type: application/json" \
  -d '{
    "term": "ai_agents",
    "volumes": [15, 45, 120],
    "window_duration_minutes": 60,
    "unique_authors": 95
  }' | jq
```

### Narrative Semantic Drift & Sentiment Shifts
Track centroid embedding drift and sentiment inversions between two time windows:
```bash
curl -s -X POST http://localhost:8000/api/v1/analytics/trends/drift \
  -H "Content-Type: application/json" \
  -d '{
    "term": "quantum",
    "baseline_texts": ["Exciting new quantum algorithms announced."],
    "current_texts": ["Severe quantum encryption flaws discovered! Critical risk."]
  }' | jq
```

### Cross-Segment Cascade Diffusion
Analyze how an information cascade spreads across demographic age groups and graph communities:
```bash
curl -s "http://localhost:8000/api/v1/analytics/diffusion/cascade_ai_news" | jq
```

### Boundary-Spanning Bridge KOLs
Identify influencers bridging disparate community clusters:
```bash
curl -s "http://localhost:8000/api/v1/analytics/network/bridge-kols?top_k=10" | jq
```

### Real-Time Server-Sent Events (SSE) Stream
Subscribe to real-time events (new posts, trend alerts, and pipeline broadcasts):
```bash
curl -N http://localhost:8000/api/v1/streaming/events
```

### Live Connector Feeds
Poll platform connectors (Twitter, Reddit, YouTube, Telegram, Bluesky):
```bash
curl -s -X POST http://localhost:8000/api/v1/connectors/bluesky/poll \
  -H "Content-Type: application/json" \
  -d '{"query": "machine learning", "limit": 5}' | jq
```

---

## Running the Automated Test Suite

Run all **146 unit and integration tests** across the entire repository:

```bash
uv run pytest -v
```

---

## Project Structure

```
HypeSignal/
├── Dataset/                     # Offline benchmark datasets (Geo, Lerman, TweetEval)
├── data/                        # Persistent DuckDB & vector files
├── scripts/
│   └── ingest_offline_datasets.py # Persistent DuckDB data ingestion utility
├── src/hypesignal/
│   ├── api/                     # FastAPI backend
│   │   ├── routes/              # Modular REST routers (timeline, sentiment, analytics, etc.)
│   │   ├── app.py               # Application factory, lifespan, CORS & error handling
│   │   ├── deps.py              # Dependency providers
│   │   ├── schemas.py           # Pydantic request/response schemas
│   │   └── streaming.py         # Resilient SSE event broadcaster with backpressure
│   ├── config.py                # Environment auto-loader (.env) and credential management
│   ├── connectors/              # Platform connectors (Twitter, Reddit, YouTube, Telegram, Bluesky)
│   ├── demographics/            # Demographics, age classification & influencer profiling
│   ├── ingestion/               # Offline benchmark dataset adapters (Geo, Lerman)
│   ├── models/                  # Canonical Pydantic schemas & enums
│   ├── network/                 # Link analysis, bridge KOLs, cascade diffusion & graph store
│   ├── nlp/                     # Sentiment, nuanced emotions, irony, stance & thread dynamics
│   ├── orchestration/           # Background tick orchestrator & async pipeline
│   ├── storage/                 # DuckDB columnar manager & Qdrant vector store
│   ├── timeline/                # Chronological query engine & conversation thread manager
│   └── trends/                  # Kinematic forecaster, narrative drift, burst & topic modelers
├── tests/                       # Automated test suite (146 passing tests)
├── .env.example                 # Environment configuration template
├── pyproject.toml               # Python 3.14 dependencies managed with UV
├── README.md                    # Getting started & execution guide
└── SYSTEM_OVERVIEW.md           # Comprehensive architectural documentation
```
