# HypeSignal

**HypeSignal** is an offline-first, AI-driven Social Media Analytics Framework. It processes multi-platform social feeds and benchmark corpora to deliver audience insights across five core analytical pillars:

1. **Continuous Timeline Management**: Unified canonical schema and chronological query engine backed by DuckDB.
2. **Multi-Dimensional Sentiment & Emotion**: Zero-shot Cardiff RoBERTa sentiment polarity, nuanced emotion detection (joy, anger, sadness, fear, surprise), irony/sarcasm inversion, and target stance classification.
3. **Automated Demographic Profiling**: Content-based geolocation extraction (spaCy NER), bio-persona interest clustering (Sentence-Transformers + Qdrant), language identification, and circadian behavioral tiers.
4. **Real-Time Trend & Dynamic Topic Detection**: Dual-tier architecture combining instant statistical frequency/Z-score burst alerts with dynamic temporal topic modeling (BERTopic).
5. **Link Analysis & Network Topology**: Follower graph topology, Key Opinion Leader (KOL) ranking (PageRank/centrality), community detection (Louvain/LPA), and information diffusion cascade tracing (structural virality).

For an in-depth architectural breakdown of every engine, refer to [SYSTEM_OVERVIEW.md](SYSTEM_OVERVIEW.md).

---

## Quick Start

### 1. Prerequisites & Environment Setup

HypeSignal uses **Python 3.14** and is managed via **UV**:

```bash
# Install dependencies into virtual environment
uv sync
```

### 2. Ingesting Data into Persistent DuckDB

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
HYPESIGNAL_DB_PATH="data/hypesignal.duckdb" uv run uvicorn hypesignal.api:app --host 0.0.0.0 --port 8080 --reload
```

#### In-Memory Mode (Stateless / Fresh Start)
If no database path is specified, HypeSignal boots with an empty in-memory store (`:memory:`):
```bash
uv run hypesignal
```

---

### 2. Launching the Frontend Dashboard

The HypeSignal frontend is an accessible, light-mode dashboard designed with dedicated navigation menus for each analytical pillar:

```bash
cd frontend
npm install
npm run dev
```

The frontend will be available at:
- **Dashboard UI**: [http://localhost:3000](http://localhost:3000)

Features available across the dedicated menus:
1. 📊 **Executive Pulse**: High-level cross-pillar summary cards and activity volume graphs.
2. 💬 **Conversations & Timeline**: Chronological post slices, multi-platform filters (X, Telegram, YouTube, Reddit, Bluesky), and conversation tree reconstruction with reply latency.
3. 🎭 **Audience Feelings & Tone**: Nuanced emotion radar, sarcasm/irony inversion, temporal sentiment tracking, and interactive live sentiment tester.
4. 👥 **Audience & Demographics**: Inferred age distributions, global geography (countries & cities), language breakdown, professional personas, and privacy-shielded ($k$-anonymity) follower audience inspector.
5. 📈 **Rising Trends & Virality**: Statistical Z-score burst radar, kinematics forecaster (velocity, acceleration, 6-state lifecycle), dynamic BERTopic clusters, and narrative drift tracker.
6. 🕸️ **Network & Viral Spread**: Follower topology visualizer, Key Opinion Leader (KOL) ranking, boundary-spanning bridge connectors, and cross-segment cascade diffusion paths.
7. 📡 **Live Feeds & Sources**: Operational status and rate limit monitoring for all 5 platform connectors, on-demand polling with DuckDB ingestion, and real-time SSE stream ticker.

---

## Interactive API Documentation

Once the backend server is running on port 8080, navigate to:
- **Swagger UI**: [http://localhost:8080/docs](http://localhost:8080/docs)
- **ReDoc**: [http://localhost:8080/redoc](http://localhost:8080/redoc)
- **Health Check**: [http://localhost:8080/health](http://localhost:8080/health)

---

## Example API Queries

### Health Check
```bash
curl -s http://localhost:8080/health | jq
```

### Multi-Dimensional Sentiment Analysis
Zero-shot sentiment polarity, emotion probabilities, sarcasm inversion, and stance:
```bash
curl -s -X POST http://localhost:8080/api/v1/sentiment/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "text": "I am thrilled and ecstatic about the breakthrough research in foundation models! #AI"
  }' | jq
```

### Demographic & Persona Profiling
Extract geographic entities, persona classification, and behavioral patterns:
```bash
curl -s -X POST http://localhost:8080/api/v1/demographics/profile-user \
  -H "Content-Type: application/json" \
  -d '{
    "screen_name": "ai_researcher",
    "bio": "Senior ML architect building transformer models in San Francisco, CA",
    "location_raw": "San Francisco, CA"
  }' | jq
```

### Audience Breakdown (Aggregate Demographics)
```bash
curl -s "http://localhost:8080/api/v1/demographics/breakdown?limit=500" | jq
```

### Key Opinion Leader (KOL) Leaderboard
Retrieve ranked influencers with community cluster IDs and composite centrality scores:
```bash
curl -s "http://localhost:8080/api/v1/network/kols?top_k=10" | jq
```

### Real-Time Trend & Burst Radar
```bash
curl -s "http://localhost:8080/api/v1/trends/overview?window_duration_minutes=60" | jq
```

### Live Connector Feeds
Poll platform connector stubs (Twitter, Reddit, YouTube, Telegram, Bluesky):
```bash
curl -s -X POST http://localhost:8080/api/v1/connectors/bluesky/poll \
  -H "Content-Type: application/json" \
  -d '{"query": "quantum computing", "limit": 2}' | jq
```

---

## Running the Automated Test Suite

Run all 146 unit and integration tests across the entire repository:

```bash
uv run pytest
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
│   │   ├── routes/              # Modular REST routers (timeline, sentiment, etc.)
│   │   ├── app.py               # Application factory & lifespan
│   │   ├── deps.py              # Dependency providers
│   │   └── schemas.py           # Pydantic request/response schemas
│   ├── connectors/              # Platform ingestion stubs (Twitter, Reddit, YouTube, TG)
│   ├── demographics/            # Demographics & persona profiling engine
│   ├── ingestion/               # Offline dataset adapters (Geo, Lerman)
│   ├── models/                  # Canonical Pydantic schemas & enums
│   ├── network/                 # Link analysis, KOL ranking, cascade tracer
│   ├── nlp/                     # Sentiment, emotion, irony, stance engines
│   ├── storage/                 # DuckDB & Qdrant vector managers
│   ├── timeline/                # Chronological query & replay manager
│   └── trends/                  # Statistical burst detector & BERTopic modeler
├── tests/                       # Automated test suite (85 passing tests)
├── pyproject.toml               # Python 3.14 dependencies managed with UV
├── README.md                    # Getting started & execution guide
└── SYSTEM_OVERVIEW.md           # Comprehensive architectural documentation
```
