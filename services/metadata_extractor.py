import os
import time
from typing import List, Dict, Any, Optional
from loguru import logger

from utils.helpers import parse_json_response
from utils.models import DocumentType, Stance, ArgumentCategory

try:
    from langchain_groq import ChatGroq
    _HAS_CHATGROQ = True
except Exception:
    _HAS_CHATGROQ = False


class MetadataExtractor:
    """Extracts document-level and chunk-level metadata using LLM."""

    def __init__(self, llm_client: Optional[Any] = None, batch_size: int = 10):
        self.batch_size = int(batch_size)

        if llm_client:
            self.llm = llm_client
            logger.info("MetadataExtractor: Using injected client")
        else:
            if not _HAS_CHATGROQ:
                raise RuntimeError("ChatGroq unavailable")

            self.llm = ChatGroq(
                model=os.getenv("LLM_MODEL", "llama-3.3-70b-versatile"),
                groq_api_key=os.getenv("GROQ_API_KEY"),
                temperature=float(os.getenv("LLM_TEMPERATURE", 0.0)),
            )
            logger.info("MetadataExtractor: ChatGroq client initialized")

    # DOCUMENT-LEVEL METADATA
    def extract_document_metadata(self, sample_text: str) -> Dict[str, Any]:
        logger.info("Extracting document-level metadata")

        prompt = f"""
        You are a legal-document classification expert.
        Extract the following metadata from this legal brief:

        Return ONLY valid JSON:
        {{
          "case_name": "Party v. Party or null",
          "document_type": "brief | motion | opinion | pleading | amicus_brief | other",
          "court": "Court name or null",
          "filing_date": "YYYY or full date or null"
        }}

        Text sample (first 3000 chars):
        {sample_text[:3000]}

        Return JSON only.
        """

        try:
            resp = self.llm.invoke(prompt)
            raw = getattr(resp, "content", resp)
            parsed = parse_json_response(raw)
            normalized = self._normalize_document_metadata(parsed)
            return normalized

        except Exception as e:
            logger.error(f"Document metadata extraction failed: {e}")
            return {}

    # CHUNK METADATA EXTRACTION — BATCHED
    def extract_chunk_metadata_batch(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        logger.info(f"Extracting chunk metadata for {len(chunks)} chunks… batch_size={self.batch_size}")

        enriched = []

        for i in range(0, len(chunks), self.batch_size):
            batch = chunks[i:i + self.batch_size]
            logger.info(f"Processing batch {i//self.batch_size + 1} with {len(batch)} chunks")

            try:
                batch_meta = self._extract_batch_metadata(batch)

                if isinstance(batch_meta, dict):
                    batch_meta = [batch_meta]

                # pad if necessary
                while len(batch_meta) < len(batch):
                    batch_meta.append({})

                # merge metadata
                for ch, meta in zip(batch, batch_meta):
                    ch.setdefault("metadata", {})
                    ch["metadata"].update(self._normalize_chunk_metadata(meta))
                    enriched.append(ch)

            except Exception as e:
                logger.error(f"Batch metadata extraction error: {e}")
                # default fallback
                for ch in batch:
                    ch.setdefault("metadata", {})
                    ch["metadata"].update(self._normalize_chunk_metadata({}))
                    enriched.append(ch)

            time.sleep(0.1)

        return enriched

    # INTERNAL — CHUNK BATCH PROMPT
    def _extract_batch_metadata(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not chunks:
            return []

        chunks_text = ""
        for idx, ch in enumerate(chunks):
            snippet = ch.get("text", "")[:800]
            chunks_text += f"\n\n--- CHUNK {idx} ---\n{snippet}"

        prompt = f"""
        You are a Supreme Court–class legal analyst.
        Analyze each chunk separately and extract structured metadata.

        For EACH chunk, return one JSON object with:
        {{
          "stance": "for | against | neutral | plaintiff | defendant | amicus",
          "importance_score": 0.0–1.0 (float),
          "legal_concepts": ["concept1", "concept2"],
          "argument_type": "statutory | regulatory | constitutional | case_law | procedural | policy | other"
        }}

        STANCE CLASSIFICATION RULE:
        - If text *supports/justifies* something → stance="for"
        - If text *criticizes/challenges/opposes* → stance="against"
        - If it's from plaintiff/defendant perspective → use that label
        - If amici speaking → stance="amicus"
        - If unclear → "neutral"

        Return EXACT JSON ARRAY with {len(chunks)} objects.
        Text:
        {chunks_text}
        """

        resp = self.llm.invoke(prompt)
        raw = getattr(resp, "content", resp)
        parsed = parse_json_response(raw)

        # ensure list
        if isinstance(parsed, dict):
            parsed = [parsed]

        # pad to required length
        while len(parsed) < len(chunks):
            parsed.append({})

        return parsed[:len(chunks)]

    # NORMALIZATION UTILITIES

    def _normalize_document_metadata(self, md: Dict[str, Any]) -> Dict[str, Any]:
        if not md or not isinstance(md, dict):
            return {}

        out = {}
        out["case_name"] = str(md.get("case_name")).strip() if md.get("case_name") else None

        raw_type = str(md.get("document_type", "other")).lower()
        try:
            out["document_type"] = DocumentType(raw_type)
        except Exception:
            out["document_type"] = DocumentType.OTHER

        court = md.get("court")
        out["court"] = str(court).strip() if court else None

        filing_date = md.get("filing_date")
        out["filing_date"] = str(filing_date).strip() if filing_date else None

        return out

    def _normalize_chunk_metadata(self, md: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(md, dict):
            md = {}

        out = {}

        # -------- stance --------
        stance_raw = str(md.get("stance", "neutral")).lower().strip()
        try:
            out["stance"] = Stance(stance_raw)
        except Exception:
            out["stance"] = Stance.NEUTRAL

        #Importance_score
        try:
            sc = float(md.get("importance_score", 0.5))
            out["importance_score"] = max(0.0, min(1.0, sc))
        except Exception:
            out["importance_score"] = 0.5

        #Legal concepts
        concepts = md.get("legal_concepts", [])
        if isinstance(concepts, list):
            out["legal_concepts"] = [str(c).strip() for c in concepts if c]
        else:
            out["legal_concepts"] = [c.strip() for c in str(concepts).split(",") if c.strip()]

        #Argument type
        arg_raw = str(md.get("argument_type", "other")).lower()
        try:
            out["argument_type"] = ArgumentCategory(arg_raw)
        except Exception:
            out["argument_type"] = ArgumentCategory.OTHER

        return out

