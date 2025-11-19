from typing import List, Dict, Any, Optional
from loguru import logger
from rapidfuzz import fuzz

from utils.models import FinalKeyPoint, Stance


class PostProcessor:
    """Processes extracted arguments and returns ranked, balanced final points."""

    def __init__(self, final_k: int = 10, dedup_threshold: float = 0.85):
        self.final_k = int(final_k)
        self.dedup_threshold = float(dedup_threshold)

    # MAIN ENTRY
    def process_and_rank(
        self,
        points: List[Dict[str, Any]],
        retrieval_scores: Optional[Dict[str, float]] = None
    ) -> List[FinalKeyPoint]:

        logger.info(f"PostProcessor: processing {len(points)} points.")

        if not points:
            return []

        # 1. Merge retrieval scores
        if retrieval_scores:
            points = self._merge_retrieval_scores(points, retrieval_scores)

        # 2. Compute combined scores
        points = [self._calculate_combined_score(p) for p in points]

        # 3. Deduplicate summaries
        points = self._dedupe(points)

        # 4. Balanced selection (auto-fallback)
        points = self._select_balanced(points, self.final_k)

        # 5. Convert to FinalKeyPoint Pydantic models
        final_list = self._assign_final_ranks(points)

        logger.info(f"PostProcessor: final selected = {len(final_list)}")
        return final_list

    # MERGE RETRIEVAL SCORES
    def _merge_retrieval_scores(self, points: List[Dict[str, Any]], scores: Dict[str, float]) -> List[Dict[str, Any]]:
        for p in points:
            chunk_id = (
                p.get("chunk_id") or
                p.get("metadata", {}).get("chunk_id")
            )
            if chunk_id and chunk_id in scores:
                p["retrieval_score"] = float(scores[chunk_id])
            else:
                p.setdefault("retrieval_score", 0.0)
        return points

    # SCORE COMPUTATION
    def _calculate_combined_score(self, p: Dict[str, Any]):
        importance = self._safe_float(p.get("importance_score"), default=0.5)
        retrieval = self._safe_float(p.get("retrieval_score"), default=0.0)

        combined = (importance * 0.7) + (retrieval * 0.3)

        p["importance_score"] = max(0.0, min(1.0, importance))
        p["retrieval_score"] = max(0.0, min(1.0, retrieval))
        p["combined_score"] = round(combined, 4)

        return p

    # DEDUPLICATION
    def _dedupe(self, points: List[Dict[str, Any]]):
        if len(points) <= 1:
            return points

        # sort by best score first
        pts = sorted(points, key=lambda x: x["combined_score"], reverse=True)

        kept = []
        summaries = []

        for p in pts:
            summary = p.get("summary", "").strip().lower()
            if not summary:
                kept.append(p)
                continue

            is_dup = False
            for s in summaries:
                sim = fuzz.ratio(summary, s) / 100.0
                if sim >= self.dedup_threshold:
                    is_dup = True
                    break

            if not is_dup:
                kept.append(p)
                summaries.append(summary)

        logger.info(f"Deduplicated {len(points)} → {len(kept)}")
        return kept

    # BALANCED SELECTION
    def _select_balanced(self, points: List[Dict[str, Any]], k: int):
        if len(points) <= k:
            return points

        # group by stance
        groups = {
            "for": [],
            "against": [],
            "neutral": [],
            "amicus": []
        }

        for p in points:
            s = self._norm_stance(p.get("stance"))
            groups[s].append(p)

        # sort each by score
        for g in groups.values():
            g.sort(key=lambda x: x["combined_score"], reverse=True)

        # If document is one-sided (e.g., all "against")
        non_empty_groups = [g for g in groups.values() if g]
        if len(non_empty_groups) == 1:
            # return top-k from this single stance
            return non_empty_groups[0][:k]

        # Otherwise do balanced selection
        half = k // 2

        selected = []
        selected.extend(groups["for"][:half])
        selected.extend(groups["against"][:half])

        remaining = k - len(selected)
        pool = (
            groups["for"][half:] +
            groups["against"][half:] +
            groups["amicus"] +
            groups["neutral"]
        )
        pool.sort(key=lambda x: x["combined_score"], reverse=True)
        selected.extend(pool[:remaining])

        return selected[:k]

    # FINAL RANK ASSIGNMENT
    def _assign_final_ranks(self, points: List[Dict[str, Any]]):
        out = []
        sorted_pts = sorted(points, key=lambda x: x["combined_score"], reverse=True)

        for rank, p in enumerate(sorted_pts, 1):

            stance_enum = self._convert_to_stance(p.get("stance"))

            final = FinalKeyPoint(
                summary=p.get("summary", ""),
                importance=p.get("importance"),
                importance_score=p.get("importance_score", 0.5),
                stance=stance_enum,
                supporting_quote=p.get("supporting_quote"),
                legal_concepts=p.get("legal_concepts", []),
                page_start=self._safe_int(p.get("page_start")),
                page_end=self._safe_int(p.get("page_end")),
                line_start=self._safe_int(p.get("line_start")),
                line_end=self._safe_int(p.get("line_end")),
                category=p.get("category"),
                retrieval_score=p.get("retrieval_score", 0.0),
                combined_score=p.get("combined_score", 0.0),
                final_rank=rank
            )
            out.append(final)

        return out

    # HELPERS
    def _norm_stance(self, s: Any) -> str:
        if isinstance(s, Stance):
            sval = s.value.lower()
        else:
            sval = str(s).lower().strip()

        if sval in ("plaintiff", "for"):
            return "for"
        if sval in ("defendant", "against"):
            return "against"
        if sval == "amicus":
            return "amicus"
        return "neutral"

    def _convert_to_stance(self, s: Any) -> Stance:
        try:
            if isinstance(s, Stance):
                return s
            return Stance(str(s).lower())
        except Exception:
            return Stance.NEUTRAL

    def _safe_float(self, v: Any, default=0.0):
        try:
            return float(v)
        except:
            return float(default)

    def _safe_int(self, v: Any):
        try:
            return int(v) if v is not None else None
        except:
            return None
