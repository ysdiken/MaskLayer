from datetime import datetime
from typing import Literal
from pydantic import BaseModel


class Span(BaseModel):
    start: int
    end: int
    label: str
    source: Literal["regex", "ner", "manual", "gazetteer"]
    confidence: float
    text: str
    canonical_id: str | None = None


class ManualSpanRequest(BaseModel):
    """
    A user-confirmed span. The pipeline finds every occurrence of `text`
    in the document and masks them all with the same placeholder.
    Confidence is always 1.0 (user-confirmed beats any detector).
    """
    text: str   # exact string to find and mask
    label: str  # taxonomy label, e.g. "Person", "Company", or custom "Company_Secret"


class MaskRequest(BaseModel):
    text: str
    language: str = "tr"
    mode: Literal["full", "regex", "ner"] = "full"   # ablation control
    manual_spans: list[ManualSpanRequest] = []        # user-added spans


class MaskResponse(BaseModel):
    job_id: str
    original_text: str
    masked_text: str
    spans: list[Span]
    processing_time_ms: float
    # Page layout hints — None for plain text input, populated by upload endpoint
    page_breaks: list[int] | None = None
    page_separator: str | None = None


class UnmaskRequest(BaseModel):
    job_id: str
    masked_text: str


class UnmaskResponse(BaseModel):
    job_id: str
    unmasked_text: str


class JobStatus(BaseModel):
    job_id: str
    status: Literal["pending", "processing", "complete", "failed"]
    created_at: datetime
