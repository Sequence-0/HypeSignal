"""Dynamic temporal topic modeler using BERTopic (Tier 2).

Extracts semantic clusters, topic representations (c-TF-IDF), and tracks
temporal topic drift over chronological bins.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from bertopic import BERTopic
from bertopic.vectorizers import ClassTfidfTransformer
from hdbscan import HDBSCAN
from sentence_transformers import SentenceTransformer
from sklearn.cluster import MiniBatchKMeans
from sklearn.feature_extraction.text import CountVectorizer
from umap import UMAP

from hypesignal.models.canonical import CanonicalPost
from hypesignal.models.enums import PlatformType
from hypesignal.storage.duckdb_manager import DuckDBManager
from hypesignal.trends.schemas import DynamicTopicTimeline, TopicRepresentation

logger = logging.getLogger(__name__)


class DynamicTopicModeler:
    """Tier-2 Dynamic Topic Modeler wrapping BERTopic and SentenceTransformers."""

    def __init__(
        self,
        embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        min_topic_size: int = 3,
        device: Optional[str] = None,
    ) -> None:
        """Initialize DynamicTopicModeler.
        
        Args:
            embedding_model_name: HuggingFace sentence transformer checkpoint.
            min_topic_size: Minimum documents per topic cluster.
            device: 'cuda', 'cpu', or None for auto detection.
        """
        self.embedding_model_name = embedding_model_name
        self.min_topic_size = min_topic_size
        self.device = device
        self.embedding_model = SentenceTransformer(embedding_model_name, device=self.device)
        self.topic_model: Optional[BERTopic] = None
        self.last_posts_by_topic: Dict[int, List[CanonicalPost]] = {}

    def _build_bertopic(self, n_docs: int, use_kmeans_fallback: bool = False) -> BERTopic:
        """Dynamically configure UMAP and HDBSCAN parameters based on corpus size."""
        n_neighbors = max(2, min(10, n_docs - 1))
        n_components = min(5, max(2, min(n_docs // 2, 3)))
        min_cluster = max(2, min(self.min_topic_size, n_docs // 2))
        min_samples = max(1, min_cluster // 2)

        umap_model = UMAP(
            n_neighbors=n_neighbors,
            n_components=n_components,
            min_dist=0.0,
            metric="cosine",
            init="random",
            random_state=42,
        )
        if use_kmeans_fallback:
            n_clusters = max(2, min(n_docs // 2, 5))
            cluster_model = MiniBatchKMeans(n_clusters=n_clusters, random_state=42)
        else:
            cluster_model = HDBSCAN(
                min_cluster_size=min_cluster,
                min_samples=min_samples,
                cluster_selection_epsilon=0.5,
                metric="euclidean",
                cluster_selection_method="eom",
                prediction_data=True,
            )
        vectorizer_model = CountVectorizer(
            stop_words="english",
            min_df=1,
            ngram_range=(1, 2),
        )
        ctfidf_model = ClassTfidfTransformer(reduce_frequent_words=True)

        return BERTopic(
            embedding_model=self.embedding_model,
            umap_model=umap_model,
            hdbscan_model=cluster_model,
            vectorizer_model=vectorizer_model,
            ctfidf_model=ctfidf_model,
            verbose=False,
        )

    def fit_topics(
        self,
        docs: List[str],
        timestamps: Optional[List[datetime]] = None,
        nr_bins: int = 5,
    ) -> Tuple[List[TopicRepresentation], Optional[List[DynamicTopicTimeline]]]:
        """Fit BERTopic on documents and optionally track temporal topic drift.
        
        Args:
            docs: List of text documents.
            timestamps: Optional list of timestamps corresponding to each document.
            nr_bins: Number of temporal bins for topics-over-time evolution.
            
        Returns:
            Tuple of (topic_representations, dynamic_timelines).
        """
        clean_docs = [d.strip() for d in docs if d and d.strip()]
        if not clean_docs:
            return [], None

        if len(clean_docs) < 3:
            # Fallback for micro-datasets where clustering is mathematically degenerate
            fallback_rep = TopicRepresentation(
                topic_id=0,
                name="0_general_social_media",
                top_words=[("post", 1.0), ("discussion", 0.8)],
                doc_count=len(clean_docs),
                representative_docs=clean_docs[:3],
            )
            return [fallback_rep], None

        self.topic_model = self._build_bertopic(len(clean_docs))
        topics, _ = self.topic_model.fit_transform(clean_docs)

        # Check if HDBSCAN labeled all documents as noise (-1)
        topic_info = self.topic_model.get_topic_info()
        non_noise_topics = [
            int(r["Topic"]) for _, r in topic_info.iterrows() if int(r["Topic"]) != -1
        ]
        if not non_noise_topics and len(clean_docs) >= 3:
            logger.info("HDBSCAN produced only noise clusters; retrying with KMeans clustering fallback.")
            self.topic_model = self._build_bertopic(len(clean_docs), use_kmeans_fallback=True)
            topics, _ = self.topic_model.fit_transform(clean_docs)
            topic_info = self.topic_model.get_topic_info()

        # Extract Topic Representations
        representations: List[TopicRepresentation] = []

        for _, row in topic_info.iterrows():
            topic_id = int(row["Topic"])
            # Topic -1 is outliers/noise in BERTopic
            if topic_id == -1:
                continue

            name = str(row["Name"])
            count = int(row["Count"])

            # Extract top words with c-TF-IDF scores
            top_words_raw = self.topic_model.get_topic(topic_id) or []
            top_words = [(str(w), float(round(s, 4))) for w, s in top_words_raw[:10]]

            # Representative docs
            rep_docs = self.topic_model.get_representative_docs(topic_id) or []
            rep_docs_clean = [str(d) for d in rep_docs[:3]]

            representations.append(
                TopicRepresentation(
                    topic_id=topic_id,
                    name=name,
                    top_words=top_words,
                    doc_count=count,
                    representative_docs=rep_docs_clean,
                )
            )

        if not representations and clean_docs:
            logger.warning("No discrete topic clusters identified; generating fallback topic representation.")
            cv = CountVectorizer(stop_words="english", max_features=10)
            try:
                cv_fit = cv.fit(clean_docs)
                words = [(w, 1.0) for w in cv_fit.get_feature_names_out()[:10]]
            except Exception:
                words = [("post", 1.0), ("discussion", 0.8)]
            fallback_rep = TopicRepresentation(
                topic_id=0,
                name="0_general_discussion",
                top_words=words,
                doc_count=len(clean_docs),
                representative_docs=clean_docs[:3],
            )
            representations.append(fallback_rep)

        # Dynamic Topics Over Time
        timelines: Optional[List[DynamicTopicTimeline]] = None

        if timestamps and len(timestamps) == len(clean_docs) and representations:
            try:
                # Convert timestamps to pandas format
                tot_df = self.topic_model.topics_over_time(
                    docs=clean_docs,
                    timestamps=timestamps,
                    nr_bins=nr_bins,
                    evolution_tuning=True,
                    global_tuning=True,
                )

                topic_name_map = {r.topic_id: r.name for r in representations}
                topic_timelines_map: Dict[int, DynamicTopicTimeline] = {}

                for _, row in tot_df.iterrows():
                    tid = int(row["Topic"])
                    if tid == -1:
                        continue

                    t_name = topic_name_map.get(tid, f"topic_{tid}")
                    ts_val = row["Timestamp"]
                    if isinstance(ts_val, pd.Timestamp):
                        dt_val = ts_val.to_pydatetime()
                    elif isinstance(ts_val, datetime):
                        dt_val = ts_val
                    else:
                        dt_val = datetime.now(timezone.utc)

                    freq = int(row["Frequency"])
                    words_str = str(row["Words"])
                    words_list = [w.strip() for w in words_str.split(",") if w.strip()]

                    if tid not in topic_timelines_map:
                        topic_timelines_map[tid] = DynamicTopicTimeline(
                            topic_id=tid,
                            topic_name=t_name,
                            timestamps=[],
                            frequencies=[],
                            evolving_words=[],
                        )

                    topic_timelines_map[tid].timestamps.append(dt_val)
                    topic_timelines_map[tid].frequencies.append(freq)
                    topic_timelines_map[tid].evolving_words.append(words_list)

                timelines = list(topic_timelines_map.values())
            except Exception as e:
                logger.warning("Topics over time calculation encountered an issue: %s", e)
                timelines = None

        return representations, timelines

    def fit_from_duckdb(
        self,
        db: DuckDBManager,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 1000,
        nr_bins: int = 5,
    ) -> Tuple[List[TopicRepresentation], Optional[List[DynamicTopicTimeline]]]:
        """Load posts from DuckDB and extract dynamic topics and temporal evolution.
        
        Args:
            db: DuckDBManager instance.
            start_time: Optional lower timestamp bound.
            end_time: Optional upper timestamp bound.
            limit: Maximum posts to analyze.
            nr_bins: Number of temporal bins.
            
        Returns:
            Tuple of (topic_representations, dynamic_timelines).
        """
        where_clauses = []
        params = []

        if start_time is not None:
            where_clauses.append("timestamp >= ?")
            params.append(start_time)
        if end_time is not None:
            where_clauses.append("timestamp <= ?")
            params.append(end_time)

        where_sql = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""
        query = f"""
            SELECT id, platform, author_id, text, timestamp
            FROM posts
            {where_sql}
            ORDER BY timestamp ASC
            LIMIT ?;
        """
        params.append(int(limit))
        rows = db.con.execute(query, params).fetchall()
        if not rows:
            self.last_posts_by_topic = {}
            return [], None

        posts: List[CanonicalPost] = []
        for r in rows:
            plat_val = r[1]
            try:
                ptype = PlatformType(plat_val)
            except (ValueError, KeyError):
                ptype = PlatformType.TWITTER
            posts.append(
                CanonicalPost(
                    id=str(r[0]),
                    platform=ptype,
                    author_id=str(r[2]),
                    text=str(r[3]),
                    timestamp=r[4],
                )
            )

        docs = [p.text for p in posts]
        timestamps = [p.timestamp for p in posts]
        representations, timelines = self.fit_topics(docs=docs, timestamps=timestamps, nr_bins=nr_bins)

        # Populate posts_by_topic mapping for downstream trend ranking
        self.last_posts_by_topic = {}
        if self.topic_model and hasattr(self.topic_model, "topics_") and self.topic_model.topics_ is not None:
            for post, tid in zip(posts, self.topic_model.topics_):
                if tid != -1:
                    self.last_posts_by_topic.setdefault(int(tid), []).append(post)

        return representations, timelines
