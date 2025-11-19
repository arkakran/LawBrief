from typing import List, Dict, Any, Optional
import os
from loguru import logger

# Load environment variables FIRST
from dotenv import load_dotenv
load_dotenv()

from utils.helpers import parse_json_response
from utils.models import ExtractedPoint, LLMAnalysisOutput, Stance

# Optional import for default client
try:
    from langchain_groq import ChatGroq  # type: ignore
    _HAS_CHATGROQ = True
except Exception:
    _HAS_CHATGROQ = False


class LLMAnalyzer:
    """Analyzes retrieved chunks and extracts key arguments via LLM prompts."""

    def __init__(self, llm_client: Optional[Any] = None, max_chunks: int = 30):
        self.max_chunks = int(max_chunks)

        if llm_client is not None:
            self.llm = llm_client
            logger.info("LLMAnalyzer: using injected LLM client")
        else:
            if not _HAS_CHATGROQ:
                raise RuntimeError("langchain_groq is not available.")

            self.llm = ChatGroq(
                model=os.getenv("LLM_MODEL", "llama-3.3-70b-versatile"),
                groq_api_key=os.getenv("GROQ_API_KEY"),
                temperature=float(os.getenv("LLM_TEMPERATURE", "0.0"))
            )
            logger.info("LLMAnalyzer: ChatGroq client initialized")

    # Query variation generator
    def generate_query_variations(self, user_query: str, n: int = 4) -> List[str]:
        logger.info(f"Generating up to {n} query variations for: {user_query}")

        prompt = f"""
            Generate EXACTLY {n} different search query variations for this legal question.

            Original Query: "{user_query}"

            Variations MUST:
            - reflect statutory, constitutional, case-law and policy angles
            - be short and precise
            - avoid redundancy

            Return ONLY a JSON array of {n} strings.
            """

        try:
            response = self.llm.invoke(prompt)
            raw = getattr(response, "content", response)
            parsed = parse_json_response(raw)

            if isinstance(parsed, list) and len(parsed) >= 1:
                variations = [str(x).strip() for x in parsed if isinstance(x, str)][:n]
                if variations:
                    logger.info(f"Generated {len(variations)} variations")
                    return variations

            return [user_query]

        except Exception as e:
            logger.error(f"Query variation generation failed: {e}")
            return [user_query]

    # Main analysis
    def analyze_chunks(self, chunks: List[Dict[str, Any]], query: str) -> LLMAnalysisOutput:
        logger.info(f"Analyzing {len(chunks)} chunks for query: {query}")

        context_chunks = chunks[:self.max_chunks]
        context = self._build_context(context_chunks)
        prompt = self._build_extraction_prompt(context, query)

        try:
            response = self.llm.invoke(prompt)
            raw = getattr(response, "content", response)
            parsed = parse_json_response(raw)
            analysis = self._parse_analysis_output(parsed, context_chunks)
            logger.info(f"Analysis complete: {len(analysis.extracted_points)} points extracted")
            return analysis
        except Exception as e:
            logger.error(f"Chunk analysis failed: {e}")
            return LLMAnalysisOutput(extracted_points=[], confidence=0.0)

    # Refinement
    def refine_extraction(self, initial_points: List[Dict[str, Any]], query: str) -> List[Dict[str, Any]]:
        logger.info(f"Refining {len(initial_points)} extracted points")

        points_text = ""
        for i, p in enumerate(initial_points, 1):
            points_text += (
                f"\n\n{i}. SUMMARY: {p.get('summary', '')}\n"
                f"QUOTE: {p.get('supporting_quote', '')}\n"
                f"CONCEPTS: {p.get('legal_concepts', [])}\n"
                f"SCORE: {p.get('importance_score', 0.5)}\n"
            )

        prompt = f"""
            Refine these legal arguments by:
            1. Cleaning summaries
            2. Adding precise statutory or case citations if available
            3. Keeping the same number of items
            4. KEEPING the same importance_score values—DO NOT remove or change them

            Original Query: "{query}"

            Arguments:
            {points_text}

            Return a CLEAN JSON array with objects containing:
            - summary
            - legal_concepts
            - supporting_quote
            - importance_score (float, unchanged)

            Return JSON only.
            """

        try:
            response = self.llm.invoke(prompt)
            raw = getattr(response, "content", response)
            refined = parse_json_response(raw)

            if isinstance(refined, list) and len(refined) == len(initial_points):
                return refined

            logger.warning("Refinement returned invalid format. Using original points.")
            return initial_points

        except Exception as e:
            logger.error(f"Refinement failed: {e}")
            return initial_points

    # Internal helpers
    def _build_context(self, chunks: List[Dict[str, Any]]) -> str:
        parts = []
        for ch in chunks:
            meta = ch.get("metadata", {})
            page = meta.get("page_start", meta.get("page", "?"))
            text = ch.get("text", "")[:900]
            parts.append(f"[Page {page}]\n{text}")
        return "\n\n---\n\n".join(parts)


    def _build_extraction_prompt(self, context: str, query: str) -> str:
        return f"""
                You are a Supreme Court-level legal analyst.

                Extract EXACTLY 12-15 distinct legal arguments from the excerpts.

                STRICT RULES:
                - Each argument MUST contain "importance_score"
                - importance_score MUST be a NUMERIC FLOAT between 0.0 and 1.0
                - NEVER omit importance_score
                - NEVER return it as a string
                - NEVER return null/None
                - NEVER return empty value
                - ALWAYS include a quote
                - ALWAYS include legal_concepts list

                Return JSON ONLY in this shape:

                {{
                  "extracted_points": [
                    {{
                      "summary": "...",
                      "importance": "...",
                      "importance_score": 0.42,
                      "stance": "plaintiff|defendant|amicus|neutral",
                      "supporting_quote": "...",
                      "legal_concepts": ["...", "..."],
                      "page_start": 1,
                      "page_end": 1
                    }}
                  ],
                  "confidence": 0.0
                }}

                EXCERPTS:
                {context}
                """


    def _parse_analysis_output(self, parsed: Dict[str, Any], chunks: List[Dict[str, Any]]) -> LLMAnalysisOutput:
        extracted_points = []

        if not isinstance(parsed, dict):
            logger.warning("Parsed output is not a dict")
            return LLMAnalysisOutput(extracted_points=[], confidence=0.0)

        points = parsed.get("extracted_points", [])
        if isinstance(points, dict):
            points = [points]
        if not isinstance(points, list):
            points = []

        for idx, p in enumerate(points):
            try:
                stance_raw = str(p.get("stance", "neutral")).lower()
                try:
                    stance_enum = Stance(stance_raw)
                except:
                    stance_enum = Stance.NEUTRAL

                # FIX: dynamic scoring if missing
                raw_score = p.get("importance_score")

                if raw_score is None:
                    # Spread scores from 0.1 → 0.9
                    raw_score = 0.1 + (idx / max(1, len(points))) * 0.8

                importance_score = max(0.0, min(1.0, float(raw_score)))

                quote = p.get("supporting_quote", "") or ""
                chunk_match = self._find_chunk_for_quote(quote, chunks)

                if chunk_match:
                    meta = chunk_match["metadata"]
                    line_start = meta.get("line_start")
                    line_end = meta.get("line_end")
                    page_start = meta.get("page_start")
                    page_end = meta.get("page_end")
                else:
                    line_start = p.get("line_start")
                    line_end = p.get("line_end")
                    page_start = p.get("page_start")
                    page_end = p.get("page_end")

                data = {
                    "summary": str(p.get("summary", "")).strip(),
                    "importance": p.get("importance"),
                    "importance_score": importance_score,
                    "stance": stance_enum,
                    "supporting_quote": quote,
                    "legal_concepts": p.get("legal_concepts", []),
                    "page_start": page_start,
                    "page_end": page_end,
                    "line_start": line_start,
                    "line_end": line_end,
                    "category": p.get("category"),
                    "retrieval_score": p.get("retrieval_score"),
                    "combined_score": p.get("combined_score")
                }

                extracted_points.append(ExtractedPoint(**data))

            except Exception as e:
                logger.warning(f"Failed to parse point {idx}: {e}")

        try:
            confidence = float(parsed.get("confidence", 0.7))
        except:
            confidence = 0.7

        confidence = max(0.0, min(1.0, confidence))
        return LLMAnalysisOutput(extracted_points=extracted_points, confidence=confidence)


    def _find_chunk_for_quote(self, quote: str, chunks: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        if not quote or len(quote.strip()) < 20:
            return None

        try:
            from rapidfuzz import fuzz
        except Exception:
            return None

        best_match = None
        best_score = 0
        q = quote.lower()

        for ch in chunks:
            text = ch.get("text", "").lower()
            score = fuzz.partial_ratio(q, text)
            if score > best_score:
                best_score = score
                best_match = ch

        return best_match if best_score >= 80 else None

