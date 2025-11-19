from typing import Dict, List, Callable, Any, Optional
from loguru import logger
import time

from services.pdf_processor import PDFProcessor
from services.metadata_extractor import MetadataExtractor
from services.vector_store import LocalVectorStore
from services.llm_analyzer import LLMAnalyzer
from services.post_processor import PostProcessor
from utils.helpers import format_page_citation


class AnalysisPipeline:

    def __init__(
        self,
        final_k: int = 10,
        chunk_size: int = 1500,
        top_k_retrieval: int = 60,
        pdf_processor: Optional[PDFProcessor] = None,
        metadata_extractor: Optional[MetadataExtractor] = None,
        vector_store: Optional[LocalVectorStore] = None,
        llm_analyzer: Optional[LLMAnalyzer] = None,
        post_processor: Optional[PostProcessor] = None
    ):
        # Config
        self.final_k = int(final_k)
        self.chunk_size = int(chunk_size)
        self.top_k_retrieval = int(top_k_retrieval)

        # Services (allow injection)
        self.pdf_processor = pdf_processor or PDFProcessor(chunk_size=self.chunk_size)
        self.metadata_extractor = metadata_extractor or MetadataExtractor()
        self.vector_store = vector_store or LocalVectorStore()
        self.llm_analyzer = llm_analyzer or LLMAnalyzer()
        self.post_processor = post_processor or PostProcessor(final_k=self.final_k)

    def analyze(
        self,
        pdf_path: str,
        query: str,
        progress_callback: Optional[Callable[[str, int], None]] = None
    ) -> Dict[str, Any]:
        """
        Run the end-to-end analysis.

        Args:
            pdf_path: path to uploaded PDF file
            query: user's extraction query
            progress_callback: optional function(stage: str, percent: int)

        Returns:
            dict with keys:
              - document_metadata
              - document_id
              - query
              - total_chunks
              - key_points (list of dicts)
              - confidence (0.0-1.0)
              - timings (optional)
        """
        start_time = time.time()
        def _report(stage: str, pct: int):
            if progress_callback:
                try:
                    progress_callback(stage, int(pct))
                except Exception:
                    logger.debug("Progress callback raised an exception; ignoring.")

        logger.info(f"Starting analysis pipeline for {pdf_path}")
        _report("starting", 0)

        # Step 1: PDF -> chunks
        _report("processing_pdf", 5)
        try:
            chunks, doc_metadata, doc_id = self.pdf_processor.process_pdf(pdf_path)
        except Exception as e:
            logger.error(f"PDF processing failed: {e}")
            raise

        total_pages = doc_metadata.get("total_pages", 0)
        _report("processing_pdf", 15)

        # Step 2: Document-level metadata (LLM)
        _report("doc_metadata", 20)
        try:
            sample_text = self.pdf_processor.get_sample_text(pdf_path, num_pages=3)
            llm_doc_meta = self.metadata_extractor.extract_document_metadata(sample_text)
            # Merge
            doc_metadata.update(llm_doc_meta or {})
        except Exception as e:
            logger.error(f"Document metadata extraction error: {e}")
            # continue with basic metadata

        _report("doc_metadata", 30)

        # Step 3: Chunk-level metadata enrichment (LLM)
        _report("chunk_metadata", 35)
        try:
            enriched_chunks = self.metadata_extractor.extract_chunk_metadata_batch(chunks)
        except Exception as e:
            logger.error(f"Chunk metadata extraction failed: {e}")
            # fallback: use original chunks with default metadata
            enriched_chunks = []
            for c in chunks:
                c.setdefault("metadata", {})
                enriched_chunks.append(c)

        _report("chunk_metadata", 45)

        # Step 4: Vector store creation and retrieval
        _report("indexing", 50)
        try:
            self.vector_store.create_index(enriched_chunks)
        except Exception as e:
            logger.error(f"Vector store creation failed: {e}")
            raise

        _report("indexing", 60)

        # Generate query variations
        try:
            query_variations = self.llm_analyzer.generate_query_variations(query)
            if not query_variations:
                query_variations = [query]
        except Exception as e:
            logger.error(f"Query variation generation failed: {e}")
            query_variations = [query]

        _report("retrieval", 65)

        # Retrieve using multiple queries
        try:
            top_k_per_query = max(1, self.top_k_retrieval // max(1, len(query_variations)))
            retrieved_chunks = self.vector_store.search_multiple_queries(
                queries=query_variations,
                top_k_per_query=top_k_per_query
            )
        except Exception as e:
            logger.error(f"Retrieval failed: {e}")
            retrieved_chunks = []

        _report("retrieval", 75)

        # Step 5: LLM analysis to extract points
        try:
            analysis_output = self.llm_analyzer.analyze_chunks(retrieved_chunks, query)
        except Exception as e:
            logger.error(f"LLM analysis failed: {e}")
            analysis_output = None

        _report("analysis", 85)

        # If analysis failed or returned nothing, return empty response
        if not analysis_output or not getattr(analysis_output, "extracted_points", None):
            logger.warning("No analysis output; returning empty result set")
            result = {
                "document_metadata": doc_metadata,
                "document_id": doc_id,
                "query": query,
                "total_chunks": len(enriched_chunks),
                "key_points": [],
                "confidence": 0.0,
                "timings": {"elapsed_seconds": round(time.time() - start_time, 2)}
            }
            _report("done", 100)
            return result

        _report("analysis", 90)

        # Step 6: Refinement (add citations, improve summaries)
        # Convert extracted_points (pydantic models) to dicts for refinement
        initial_points = [p.dict() for p in analysis_output.extracted_points]
        try:
            refined_points = self.llm_analyzer.refine_extraction(initial_points, query)
            if not isinstance(refined_points, list) or len(refined_points) == 0:
                refined_points = initial_points
        except Exception as e:
            logger.error(f"Refinement step failed: {e}")
            refined_points = initial_points

        _report("refinement", 92)

        # Step 7: Post-process and rank
        # Build retrieval scores map: chunk_id -> score
        retrieval_scores: Dict[str, float] = {}
        for ch in retrieved_chunks:
            meta = ch.get("metadata", {})
            chunk_id = meta.get("chunk_id")
            if chunk_id:
                retrieval_scores[chunk_id] = max(retrieval_scores.get(chunk_id, 0.0), float(ch.get("score", 0.0)))

        try:
            final_points_models = self.post_processor.process_and_rank(refined_points, retrieval_scores)
        except Exception as e:
            logger.error(f"Post-processing failed: {e}")
            final_points_models = []

        _report("post_processing", 97)

        # Convert FinalKeyPoint models to plain dicts (safe conversion)
        key_points = []
        for p in final_points_models:
            try:
                # p is expected to be a pydantic model with .dict()
                if hasattr(p, "dict"):
                    kd = p.dict()
                else:
                    # fallback if it's a dict already
                    kd = dict(p)
                # Add citation string if page/line info present
                citation = format_page_citation(
                    kd.get("page_start") or 0,
                    kd.get("page_end") or 0,
                    kd.get("line_start"),
                    kd.get("line_end")
                )
                kd["citation"] = citation
                key_points.append(kd)
            except Exception as e:
                logger.warning(f"Failed to convert final point to dict: {e}")
                continue

        # Step 8: Confidence calculation
        try:
            all_scores = [ch.get("score", 0.0) for ch in retrieved_chunks] or [0.0]
            confidence = self._calculate_confidence(final_points_models, all_scores)
        except Exception as e:
            logger.error(f"Confidence calculation failed: {e}")
            confidence = 0.0

        elapsed = round(time.time() - start_time, 2)
        result = {
            "document_metadata": doc_metadata,
            "document_id": doc_id,
            "query": query,
            "total_chunks": len(enriched_chunks),
            "key_points": key_points,
            "confidence": confidence,
            "timings": {"elapsed_seconds": elapsed}
        }

        _report("done", 100)
        logger.info(f"Analysis complete in {elapsed}s — found {len(key_points)} key points")
        return result

    # Utilities
    def _calculate_confidence(self, points: List[Any], retrieval_scores: List[float]) -> float:
        """
        Calculate overall confidence from importance scores and retrieval quality.

        - importance_avg: average importance_score across points
        - retrieval_avg: average of top retrieval scores (top 10)
        - weights: importance 0.6, retrieval 0.4
        """
        if not points:
            return 0.0

        # points may be pydantic models or dicts
        importance_vals = []
        for p in points:
            try:
                if hasattr(p, "importance_score"):
                    importance_vals.append(float(getattr(p, "importance_score", 0.0)))
                elif isinstance(p, dict):
                    importance_vals.append(float(p.get("importance_score", 0.0)))
                else:
                    importance_vals.append(0.0)
            except Exception:
                importance_vals.append(0.0)

        importance_avg = sum(importance_vals) / max(1, len(importance_vals))

        # retrieval scores: use top 10 if available
        top_scores = sorted([float(s or 0.0) for s in retrieval_scores], reverse=True)[:10]
        retrieval_avg = (sum(top_scores) / len(top_scores)) if top_scores else 0.0

        confidence = (importance_avg * 0.6) + (retrieval_avg * 0.4)
        return round(max(0.0, min(1.0, confidence)), 2)

