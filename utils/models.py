from pydantic import BaseModel, Field
from typing import List, Optional
from enum import Enum

class DocumentType(str, Enum):
    BRIEF = "brief"
    MOTION = "motion"
    OPINION = "opinion"
    PLEADING = "pleading"
    AMICUS_BRIEF = "amicus_brief"
    OTHER = "other"

class Stance(str, Enum):
    PLAINTIFF = "plaintiff"
    DEFENDANT = "defendant"
    AMICUS = "amicus"
    FOR = "for"
    AGAINST = "against"
    NEUTRAL = "neutral"
    UNKNOWN = "unknown"

class ArgumentCategory(str, Enum):
    STATUTORY = "statutory"
    REGULATORY = "regulatory"
    CONSTITUTIONAL = "constitutional"
    CASE_LAW = "case_law"
    PROCEDURAL = "procedural"
    POLICY = "policy"
    OTHER = "other"

class ExtractedPoint(BaseModel):
    summary: str = Field(..., min_length=5)
    importance: Optional[str] = None
    importance_score: float = Field(..., ge=0.0, le=1.0)

    stance: Stance = Field(default=Stance.NEUTRAL)

    supporting_quote: Optional[str] = None
    legal_concepts: List[str] = []

    page_start: Optional[int] = None
    page_end: Optional[int] = None
    line_start: Optional[int] = None
    line_end: Optional[int] = None

    category: Optional[ArgumentCategory] = None

    retrieval_score: Optional[float] = None
    combined_score: Optional[float] = None


class FinalKeyPoint(ExtractedPoint):
    final_rank: int = Field(..., ge=1)


class LLMAnalysisOutput(BaseModel):
    extracted_points: List[ExtractedPoint]
    confidence: float = Field(..., ge=0.0, le=1.0)

