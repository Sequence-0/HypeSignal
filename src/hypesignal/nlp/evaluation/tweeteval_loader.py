"""Isolated loader and benchmark evaluator for TweetEval datasets.

TweetEval is kept strictly isolated from the analytical DuckDB timeline,
serving as the ground-truth benchmark for calibrating and evaluating
sentiment, multi-class emotion, irony/sarcasm, and stance detection models.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkSample:
    """A single labeled sample from a benchmark dataset."""
    text: str
    label_id: int
    label_name: str
    task: str
    split: str


class TweetEvalLoader:
    """Loader for isolated TweetEval multi-task benchmark splits."""

    TASKS = ["emotion", "irony", "sentiment", "hate", "offensive", "emoji", "stance"]

    def __init__(
        self,
        base_dir: Union[str, Path] = "Dataset/twitter/TweetEval(Sentiment)/datasets",
    ) -> None:
        """Initialize loader pointing to TweetEval dataset directory."""
        self.base_dir = Path(base_dir)
        if not self.base_dir.exists():
            raise FileNotFoundError(f"TweetEval directory not found: {self.base_dir}")

    def load_mapping(self, task: str) -> Dict[int, str]:
        """Load integer label to human-readable string mapping.
        
        Args:
            task: Task name (e.g. 'emotion', 'irony', 'sentiment', 'stance').
        """
        task_dir = self.base_dir / task.split("/")[0]
        mapping_file = task_dir / "mapping.txt"
        if not mapping_file.exists():
            raise FileNotFoundError(f"Mapping file not found: {mapping_file}")

        mapping: Dict[int, str] = {}
        with open(mapping_file, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 2:
                    try:
                        mapping[int(parts[0])] = parts[1].strip()
                    except ValueError:
                        continue
        return mapping

    def load_split(
        self,
        task: str,
        split: str = "test",
        limit: Optional[int] = None,
    ) -> List[BenchmarkSample]:
        """Load labeled texts for a specific task and split.
        
        Args:
            task: Task name (e.g. 'emotion', 'irony', 'sentiment', or 'stance/climate').
            split: 'train', 'val', or 'test'.
            limit: Maximum samples to load.
            
        Returns:
            List of BenchmarkSample instances.
        """
        if "/" in task:
            main_task, sub_task = task.split("/", 1)
            task_path = self.base_dir / main_task / sub_task
            mapping = self.load_mapping(main_task)
        else:
            task_path = self.base_dir / task
            mapping = self.load_mapping(task)

        text_file = task_path / f"{split}_text.txt"
        label_file = task_path / f"{split}_labels.txt"

        if not text_file.exists() or not label_file.exists():
            raise FileNotFoundError(f"Split files not found for task '{task}' ({split}): {text_file}")

        samples: List[BenchmarkSample] = []
        with open(text_file, "r", encoding="utf-8", errors="replace") as ft, \
             open(label_file, "r", encoding="utf-8", errors="replace") as fl:
            for text_line, label_line in zip(ft, fl):
                if limit and len(samples) >= limit:
                    break

                text = text_line.strip()
                label_str = label_line.strip()
                if not label_str:
                    continue

                try:
                    label_id = int(label_str)
                except ValueError:
                    continue

                label_name = mapping.get(label_id, f"label_{label_id}")
                samples.append(
                    BenchmarkSample(
                        text=text,
                        label_id=label_id,
                        label_name=label_name,
                        task=task,
                        split=split,
                    )
                )

        logger.info("Loaded %d %s samples for task '%s'", len(samples), split, task)
        return samples

    def get_task_summary(self, task: str) -> Dict[str, int]:
        """Count number of samples available across train, val, and test splits."""
        summary = {}
        for split in ["train", "val", "test"]:
            try:
                samples = self.load_split(task, split=split)
                summary[split] = len(samples)
            except FileNotFoundError:
                summary[split] = 0
        return summary
