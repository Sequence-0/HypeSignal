"""Dataset ingestion adapters for HypeSignal."""

from hypesignal.ingestion.base import BaseDatasetAdapter
from hypesignal.ingestion.geo_adapter import GeoDatasetAdapter
from hypesignal.ingestion.lerman_adapter import LermanDatasetAdapter

__all__ = ["BaseDatasetAdapter", "GeoDatasetAdapter", "LermanDatasetAdapter"]
