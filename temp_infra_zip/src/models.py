from dataclasses import dataclass
from datetime import datetime
from typing import Optional


@dataclass
class SensitiveModel:
    """Represents a sensitive model with JavaScript monitoring program."""
    model_id: str
    js_program: str
    uploader_username: str
    created_at: str
    updated_at: str


@dataclass
class DownloadHistoryEntry:
    """Represents a download event for a sensitive model."""
    model_id: str
    downloader_username: str
    download_timestamp: str
    success: bool
    error_message: Optional[str] = None
