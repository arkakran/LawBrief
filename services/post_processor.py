from typing import List, Dict, Any, Optional
from loguru import logger

from rapidfuzz import fuzz

from utils.models import FinalKeyPoint, Stance


class PostProcessor:
    """Processes extracted arguments and returns ranked, balanced final points."""

    def __init__(self, final_k: int = 10, dedup_threshold: float = 0.85):
        self.final_k = int(final_k)
        self.dedup_threshold = float(dedup_threshold)

    def process_and_rank(
        self,
        points: List[Dict[str, Any]],
        retrieval_scores: Optional[Dict[str, float]] = None
    ) -> List[FinalKeyPoint]:

        logger.info(f"PostProcessor: processing {len(points)} points (final_k={self.final_k})")

        if not points:
            return []

        # Merge retrieval scores (if provided)
        if retrieval_scores:
            points = self._merge_retrieval_scores(points, retrieval_scores)

        # Compute combined scores
        scored = [self._calculate_combined_score(p) for p in points]

        # Deduplicate by summary text
        deduped = self._deduplicate_points(scored, threshold=self.dedup_threshold)

        # Select balanced top-k
        selected = self._select_balanced_top_k(deduped, self.final_k)

        # Assign final ranks and convert to FinalKeyPoint models
        final_points = self._assign_final_ranks(selected)

        logger.info(f"PostProcessor: selected {len(final_points)} final points")
        return final_points


    def _merge_retrieval_scores(self, points: List[Dict[str, Any]], scores: Dict[str, float]) -> List[Dict[str, Any]]:
        """Attach retrieval_score to each point where possible using chunk_id or metadata.chunk_id."""
        for p in points:
            chunk_id = p.get("chunk_id") or (p.get("metadata") or {}).get("chunk_id")
            if chunk_id and chunk_id in scores:
                try:
                    p["retrieval_score"] = float(scores[chunk_id])
                except Exception:
                    p["retrieval_score"] = p.get("retrieval_score", 0.0)
            else:
                p.setdefault("retrieval_score", 0.0)
        return points

    def _calculate_combined_score(self, point: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compute combined score from importance_score and retrieval_score.
        """
        importance = self._safe_float(point.get("importance_score"), default=0.5)
        retrieval = self._safe_float(point.get("retrieval_score"), default=0.0)

        combined = (importance * 0.7) + (retrieval * 0.3)
        point["combined_score"] = round(float(combined), 4)

        # Ensure importance_score present and clipped
        point["importance_score"] = round(max(0.0, min(1.0, importance)), 4)
        point["retrieval_score"] = round(max(0.0, min(1.0, retrieval)), 4)

        return point

    def _deduplicate_points(self, points: List[Dict[str, Any]], threshold: float = 0.85) -> List[Dict[str, Any]]:
        """
        Remove duplicates based on summary similarity (rapidfuzz ratio).
        Keeps the highest combined_score variant among duplicates.
        """
        if len(points) <= 1:
            return points

        # Sort by combined_score desc so keep best first
        sorted_points = sorted(points, key=lambda x: x.get("combined_score", 0.0), reverse=True)

        unique = []
        for p in sorted_points:
            summary = str(p.get("summary", "")).strip()
            if not summary:
                # Treat empty summary as unique (will likely be filtered later)
                unique.append(p)
                continue

            is_dup = False
            for u in unique:
                # Compare with already accepted summaries
                u_summary = str(u.get("summary", "")).strip()
                if not u_summary:
                    continue
                try:
                    sim = fuzz.ratio(summary.lower(), u_summary.lower()) / 100.0
                except Exception:
                    # On fuzz errors, be conservative
                    sim = 0.0

                if sim >= threshold:
                    is_dup = True
                    break

            if not is_dup:
                unique.append(p)

        logger.info(f"Deduplicated {len(points)} -> {len(unique)} points")
        return unique

    def _select_balanced_top_k(self, points: List[Dict[str, Any]], k: int) -> List[Dict[str, Any]]:
        if len(points) <= k:
            return points

        # Group by stance normalized to 'for', 'against', 'neutral'
        groups = {"for": [], "against": [], "neutral": []}

        for p in points:
            s = p.get("stance")
            normalized = self._normalize_stance(s)
            if normalized == "for":
                groups["for"].append(p)
            elif normalized == "against":
                groups["against"].append(p)
            else:
                groups["neutral"].append(p)

        # Sort each group by combined_score desc
        for key in groups:
            groups[key].sort(key=lambda x: x.get("combined_score", 0.0), reverse=True)

        selected = []
        # Target per side (for vs against)
        target_per_side = k // 2

        selected.extend(groups["for"][:target_per_side])
        selected.extend(groups["against"][:target_per_side])

        remaining_slots = k - len(selected)
        if remaining_slots > 0:
            # Build pool of remaining candidates
            pool = []
            pool.extend(groups["for"][target_per_side:])
            pool.extend(groups["against"][target_per_side:])
            pool.extend(groups["neutral"])

            pool.sort(key=lambda x: x.get("combined_score", 0.0), reverse=True)
            selected.extend(pool[:remaining_slots])

        # Ensure not exceeding k
        return selected[:k]

    def _assign_final_ranks(self, points: List[Dict[str, Any]]) -> List[FinalKeyPoint]:
        """Sort final points by combined_score and convert to FinalKeyPoint Pydantic models with ranks."""
        sorted_points = sorted(points, key=lambda x: x.get("combined_score", 0.0), reverse=True)

        final = []
        for rank, p in enumerate(sorted_points, start=1):
            try:
                # Normalize stance to Stance enum if possible
                stance_val = p.get("stance")
                try:
                    # Accept either Stance enum or string
                    if isinstance(stance_val, Stance):
                        stance_enum = stance_val
                    else:
                        stance_enum = Stance(str(stance_val).lower())
                except Exception:
                    stance_enum = Stance.NEUTRAL

                fk = FinalKeyPoint(
                    summary=str(p.get("summary", "")).strip(),
                    importance=p.get("importance"),
                    importance_score=self._safe_float(p.get("importance_score"), default=0.5),
                    stance=stance_enum,
                    supporting_quote=p.get("supporting_quote"),
                    legal_concepts=p.get("legal_concepts", []),
                    page_start=self._safe_int(p.get("page_start")),
                    page_end=self._safe_int(p.get("page_end")),
                    line_start=self._safe_int(p.get("line_start")),
                    line_end=self._safe_int(p.get("line_end")),
                    category=p.get("category"),
                    retrieval_score=self._safe_float(p.get("retrieval_score"), default=0.0),
                    combined_score=self._safe_float(p.get("combined_score"), default=0.0),
                    final_rank=rank
                )
                final.append(fk)
            except Exception as e:
                logger.error(f"Failed to build FinalKeyPoint for rank {rank}: {e}")
                continue

        return final


    def _normalize_stance(self, stance: Any) -> str:
        """Normalize various possible stance representations to 'for'|'against'|'neutral'."""
        if isinstance(stance, Stance):
            sval = stance.value.lower()
        else:
            sval = str(stance or "").lower()

        if sval in ("plaintiff", "for"):
            return "for"
        if sval in ("defendant", "against"):
            return "against"
        return "neutral"

    def _safe_float(self, v: Any, default: float = 0.0) -> float:
        try:
            return float(v)
        except Exception:
            return float(default)

    def _safe_int(self, v: Any) -> Optional[int]:
        try:
            if v is None:
                return None
            return int(v)
        except Exception:
            return None

