import os
import time
from typing import List, Dict, Any, Optional
from loguru import logger

from utils.helpers import parse_json_response
from utils.models import DocumentType, Stance, ArgumentCategory

try:
    from langchain_groq import ChatGroq  # type: ignore
    _HAS_CHATGROQ = True
except Exception:
    _HAS_CHATGROQ = False


class MetadataExtractor:

    def __init__(self, llm_client: Optional[Any] = None, batch_size: int = 10):
        self.batch_size = int(batch_size)

        if llm_client is not None:
            self.llm = llm_client
            logger.info("MetadataExtractor: Using injected LLM client")
        else:
            if not _HAS_CHATGROQ:
                raise RuntimeError(
                    "langchain_groq.ChatGroq not available and no llm_client provided. "
                    "Install or provide an LLM client."
                )
            self.llm = ChatGroq(
                model=os.getenv("LLM_MODEL", "llama-3.3-70b-versatile"),
                groq_api_key=os.getenv("GROQ_API_KEY"),
                temperature=float(os.getenv("LLM_TEMPERATURE", "0.0"))
            )
            logger.info("MetadataExtractor: ChatGroq client initialized")

    def extract_document_metadata(self, sample_text: str) -> Dict[str, Any]:
        logger.info("Extracting document-level metadata from sample text")
        prompt = f"""You are a legal document analyzer. Extract metadata from this legal brief.

                    Return ONLY valid JSON with this structure:
                    {{
                      "case_name": "Party v. Party or null",
                      "document_type": "brief|motion|opinion|pleading|amicus_brief|other",
                      "court": "Court name or null",
                      "filing_date": "Date or null"
                    }}

                    Now extract from the following sample (first 3000 chars):
                    {sample_text[:3000]}

                    Return JSON only."""
        try:
            response = self.llm.invoke(prompt)
            raw = getattr(response, "content", response)
            parsed = parse_json_response(raw)

            normalized = self._normalize_document_metadata(parsed)
            logger.info(f"Document metadata extracted: {normalized}")
            return normalized
        except Exception as e:
            logger.error(f"Document metadata extraction failed: {e}")
            return {}

    def extract_chunk_metadata_batch(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Extracts metadata for a list of chunks in batches.

        Each returned metadata dict will be merged into the chunk's 'metadata'.
        """
        logger.info(f"Extracting chunk metadata for {len(chunks)} chunks (batch_size={self.batch_size})")

        enriched = []

        # Create batches
        for i in range(0, len(chunks), self.batch_size):
            batch = chunks[i:i + self.batch_size]
            logger.info(f"Processing metadata batch {i // self.batch_size + 1}: {len(batch)} chunks")

            try:
                batch_meta = self._extract_batch_metadata(batch)
                # Ensure we return a list of metadata dicts with length == len(batch)
                if isinstance(batch_meta, dict):
                    batch_meta = [batch_meta]
                # pad if shorter
                while len(batch_meta) < len(batch):
                    batch_meta.append({})
                for chunk, meta in zip(batch, batch_meta[:len(batch)]):
                    # ensure metadata key exists
                    chunk.setdefault("metadata", {})
                    chunk["metadata"].update(self._normalize_chunk_metadata(meta))
                    enriched.append(chunk)
            except Exception as e:
                logger.error(f"Batch metadata extraction failed: {e}")
                # fallback: attach minimal defaults
                for chunk in batch:
                    chunk.setdefault("metadata", {})
                    chunk["metadata"].update(self._normalize_chunk_metadata({}))
                    enriched.append(chunk)

            # brief pause to avoid throttling (if necessary)
            time.sleep(0.1)

        logger.info(f"Completed metadata enrichment for {len(enriched)} chunks")
        return enriched

    def _extract_batch_metadata(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Internal: build prompt for a batch of chunks and parse response.

        Returns a list of metadata dicts (one per chunk).
        """
        if not chunks:
            return []

        chunks_text = ""
        for idx, chunk in enumerate(chunks):
            text = chunk.get("text", "")[:800]
            chunks_text += f"\n\n--- CHUNK {idx} ---\n{text}"

        prompt = f"""Analyze these legal text chunks and extract metadata for each.

                    Return a JSON array with {len(chunks)} objects, each having:
                    {{
                      "stance": "plaintiff|defendant|amicus|neutral",
                      "importance_score": 0.0-1.0,
                      "legal_concepts": ["concept1", "concept2"],
                      "argument_type": "statutory|regulatory|constitutional|case_law|procedural|policy|other"
                    }}

                    Chunks to analyze:
                    {chunks_text}

                    Return JSON array only (exactly {len(chunks)} objects):"""

        response = self.llm.invoke(prompt)
        raw = getattr(response, "content", response)
        parsed = parse_json_response(raw)

        # If the model returned a single object, coerce to list
        if isinstance(parsed, dict):
            parsed = [parsed]

        # pad to length
        while len(parsed) < len(chunks):
            parsed.append({})

        # normalize each metadata object
        return [self._normalize_chunk_metadata(item) for item in parsed[:len(chunks)]]

    def _normalize_document_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize and convert raw document metadata into stable types/enums."""
        normalized: Dict[str, Any] = {}

        if not metadata or not isinstance(metadata, dict):
            return normalized

        case_name = metadata.get("case_name")
        normalized["case_name"] = str(case_name).strip() if case_name else None

        doc_type_raw = metadata.get("document_type", "other")
        try:
            normalized["document_type"] = DocumentType(str(doc_type_raw).lower())
        except Exception:
            normalized["document_type"] = DocumentType.OTHER

        court = metadata.get("court")
        normalized["court"] = str(court).strip() if court else None

        filing_date = metadata.get("filing_date")
        normalized["filing_date"] = str(filing_date).strip() if filing_date else None

        return normalized

    def _normalize_chunk_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize single chunk metadata into stable types/enums with defaults."""
        normalized: Dict[str, Any] = {}
        if not metadata or not isinstance(metadata, dict):
            metadata = {}

        # stance
        stance_raw = str(metadata.get("stance", "neutral")).lower()
        try:
            normalized["stance"] = Stance(stance_raw)
        except Exception:
            normalized["stance"] = Stance.NEUTRAL

        # importance_score
        try:
            score = float(metadata.get("importance_score", 0.5))
            normalized["importance_score"] = max(0.0, min(1.0, score))
        except Exception:
            normalized["importance_score"] = 0.5

        # legal_concepts
        concepts = metadata.get("legal_concepts", [])
        if isinstance(concepts, list):
            normalized["legal_concepts"] = [str(c).strip() for c in concepts if c]
        else:
            # try to parse comma-separated string
            try:
                normalized["legal_concepts"] = [c.strip() for c in str(concepts).split(",") if c.strip()]
            except Exception:
                normalized["legal_concepts"] = []

        # argument_type
        arg_raw = str(metadata.get("argument_type", "other")).lower()
        try:
            normalized["argument_type"] = ArgumentCategory(arg_raw)
        except Exception:
            normalized["argument_type"] = ArgumentCategory.OTHER

        return normalized
