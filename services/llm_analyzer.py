from typing import List, Dict, Any, Optional
import os
from loguru import logger

from dotenv import load_dotenv
load_dotenv()

from utils.helpers import parse_json_response
from utils.models import ExtractedPoint, LLMAnalysisOutput, Stance

try:
    from langchain_groq import ChatGroq
    _HAS_CHATGROQ = True
except Exception:
    _HAS_CHATGROQ = False


class LLMAnalyzer:
    """Extracts legal arguments from retrieved chunks using LLM prompts."""

    def __init__(self, llm_client: Optional[Any] = None, max_chunks: int = 30):
        self.max_chunks = int(max_chunks)

        if llm_client is not None:
            self.llm = llm_client
            logger.info("LLMAnalyzer: using injected client")
        else:
            if not _HAS_CHATGROQ:
                raise RuntimeError("ChatGroq not available")

            self.llm = ChatGroq(
                model=os.getenv("LLM_MODEL", "llama-3.3-70b-versatile"),
                groq_api_key=os.getenv("GROQ_API_KEY"),
                temperature=float(os.getenv("LLM_TEMPERATURE", 0.0))
            )
            logger.info("LLMAnalyzer: ChatGroq initialized")

    # Query Variations
    def generate_query_variations(self, user_query: str, n: int = 4) -> List[str]:
        logger.info(f"Generating query variations for: {user_query}")

        prompt = f"""
        Generate {n} short, diverse legal search variations for the query:

        "{user_query}"

        Include statutory, constitutional, policy, and case-law phrasing.
        Return ONLY JSON array of strings.
        """

        try:
            resp = self.llm.invoke(prompt)
            raw = getattr(resp, "content", resp)
            parsed = parse_json_response(raw)

            if isinstance(parsed, list) and len(parsed):
                return [str(x) for x in parsed][:n]

        except Exception as e:
            logger.error(f"Variation generation error: {e}")

        return [user_query]

    # Chunk Analysis
    def analyze_chunks(self, chunks: List[Dict[str, Any]], query: str) -> LLMAnalysisOutput:
        logger.info(f"Analyzing {len(chunks)} chunks for query={query}")

        context = self._build_context(chunks[:self.max_chunks])
        prompt = self._build_extraction_prompt(context, query)

        try:
            resp = self.llm.invoke(prompt)
            raw = getattr(resp, "content", resp)
            parsed = parse_json_response(raw)

            analysis = self._parse_analysis_output(parsed, chunks)
            return analysis

        except Exception as e:
            logger.error(f"Chunk analysis failed: {e}")
            return LLMAnalysisOutput(extracted_points=[], confidence=0.0)

    # Refinement Step
    def refine_extraction(self, points: List[Dict[str, Any]], query: str):
        logger.info(f"Refining {len(points)} arguments")

        text_block = ""
        for i, p in enumerate(points, 1):
            text_block += (
                f"\n{i}. SUMMARY: {p.get('summary','')}"
                f"\nQUOTE: {p.get('supporting_quote','')}"
                f"\nCONCEPTS: {p.get('legal_concepts',[])}"
                f"\nSCORE: {p.get('importance_score',0.5)}\n"
            )

        prompt = f"""
        Refine and clean these legal arguments.
        - Keep the SAME number of arguments.
        - DO NOT change importance_score values.
        - Improve clarity and add precise citation text if present.
        Return JSON array ONLY.

        Arguments:
        {text_block}
        """

        try:
            resp = self.llm.invoke(prompt)
            raw = getattr(resp, "content", resp)
            parsed = parse_json_response(raw)

            if isinstance(parsed, list) and len(parsed) == len(points):
                return parsed

        except Exception as e:
            logger.error(f"Refinement failed: {e}")

        return points

    # Internal — Build Context
    def _build_context(self, chunks: List[Dict[str, Any]]) -> str:
        parts = []
        for ch in chunks:
            meta = ch.get("metadata", {})
            page = meta.get("page_start", meta.get("page", "?"))
            text = ch.get("text", "")[:1000]
            parts.append(f"[Page {page}]\n{text}")
        return "\n\n---\n\n".join(parts)

    # Extraction Prompt
    def _build_extraction_prompt(self, context: str, query: str) -> str:
        return f"""
        You are a Supreme Court-level legal analyst.

        Extract EXACTLY 12–15 key legal arguments from the text.

        ⭐ FOR EACH ARGUMENT YOU MUST OUTPUT:
        - summary
        - importance_score (0.0–1.0 float)
        - supporting_quote
        - legal_concepts (list)
        - page_start, page_end
        - stance = "for" | "against" | "neutral" | "plaintiff" | "defendant" | "amicus"

        ⭐ STANCE RULES:
        - If the argument SUPPORTS an action/position → stance="for"
        - If it OPPOSES or CRITICIZES → stance="against"
        - If neither clear → stance="neutral"
        - If the document itself argues from plaintiff/defendant/amicus perspective, reflect it.

        STRICT: Return VALID JSON ONLY in this shape:

        {{
          "extracted_points": [
            {{
              "summary": "...",
              "importance_score": 0.42,
              "stance": "for",
              "supporting_quote": "...",
              "legal_concepts": ["..."],
              "page_start": 1,
              "page_end": 1
            }}
          ],
          "confidence": 0.0
        }}

        EXCERPTS:
        {context}
        """

    # Parse model output
    def _parse_analysis_output(self, parsed: Dict[str, Any], chunks: List[Dict[str, Any]]):
        points_raw = parsed.get("extracted_points", [])
        if isinstance(points_raw, dict):
            points_raw = [points_raw]

        extracted = []

        for idx, p in enumerate(points_raw):
            try:
                # stance detection & correction
                stance_raw = str(p.get("stance", "neutral")).lower()

                # AUTO stance detection
                if stance_raw == "neutral":
                    text = (
                        (p.get("summary") or "") +
                        " " +
                        (p.get("supporting_quote") or "")
                    ).lower()

                    if any(k in text for k in ["supports", "upholds", "permits", "allows", "in favor"]):
                        stance_raw = "for"

                    elif any(k in text for k in ["violates", "contradicts", "against", "blocks", "undermines"]):
                        stance_raw = "against"

                try:
                    stance_enum = Stance(stance_raw)
                except Exception:
                    stance_enum = Stance.NEUTRAL

                # importance score fix
                raw_score = p.get("importance_score")
                if raw_score is None:
                    raw_score = 0.1 + (idx / max(1, len(points_raw))) * 0.8
                importance_score = float(max(0.0, min(1.0, raw_score)))

                # quote + citation matching
                quote = p.get("supporting_quote", "") or ""
                match = self._find_chunk_for_quote(quote, chunks)

                if match:
                    meta = match["metadata"]
                    page_start = meta.get("page_start")
                    page_end = meta.get("page_end")
                    line_start = meta.get("line_start")
                    line_end = meta.get("line_end")
                else:
                    page_start = p.get("page_start")
                    page_end = p.get("page_end")
                    line_start = p.get("line_start")
                    line_end = p.get("line_end")

                item = {
                    "summary": p.get("summary", ""),
                    "importance_score": importance_score,
                    "importance": p.get("importance"),
                    "stance": stance_enum,
                    "supporting_quote": quote,
                    "legal_concepts": p.get("legal_concepts", []),
                    "page_start": page_start,
                    "page_end": page_end,
                    "line_start": line_start,
                    "line_end": line_end,
                }

                extracted.append(ExtractedPoint(**item))

            except Exception as e:
                logger.warning(f"Failed to parse point {idx}: {e}")

        confidence = float(parsed.get("confidence", 0.7))
        return LLMAnalysisOutput(extracted_points=extracted, confidence=confidence)

    # Quote → Chunk matching
    def _find_chunk_for_quote(self, quote: str, chunks: List[Dict[str, Any]]):
        if not quote or len(quote.strip()) < 20:
            return None

        try:
            from rapidfuzz import fuzz
        except Exception:
            return None

        best = None
        best_score = 0
        q = quote.lower()

        for ch in chunks:
            txt = ch.get("text", "").lower()
            score = fuzz.partial_ratio(q, txt)
            if score > best_score:
                best_score = score
                best = ch

        return best if best_score >= 80 else None
