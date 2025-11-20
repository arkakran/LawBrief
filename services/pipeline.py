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
        self.final_k = final_k
        self.chunk_size = chunk_size
        self.top_k_retrieval = top_k_retrieval

        self.pdf_processor = pdf_processor or PDFProcessor(chunk_size=self.chunk_size)
        self.metadata_extractor = metadata_extractor or MetadataExtractor()
        self.vector_store = vector_store or LocalVectorStore()
        self.llm_analyzer = llm_analyzer or LLMAnalyzer()
        self.post_processor = post_processor or PostProcessor(final_k=self.final_k)


    def analyze(self, pdf_path: str, query: str, progress_callback: Optional[Callable[[str, int], None]] = None) -> Dict[str, Any]:
        start_time = time.time()

        def _step(stage: str, pct: int):
            if progress_callback:
                try: progress_callback(stage, pct)
                except: pass

        logger.info(f"Pipeline started for {pdf_path}")
        _step("start", 1)

        # 1. PDF → text chunks
        chunks, doc_metadata, doc_id = self.pdf_processor.process_pdf(pdf_path)
        _step("pdf_processing", 20)

        # 2. Document-level metadata (LLM)
        sample = self.pdf_processor.get_sample_text(pdf_path)
        try:
            llm_meta = self.metadata_extractor.extract_document_metadata(sample)
            doc_metadata.update(llm_meta)
        except Exception as e:
            logger.warning(f"Document-level metadata failed: {e}")
        _step("doc_metadata", 30)

        # 3. Chunk-level metadata (LLM batch)
        try:
            enriched_chunks = self.metadata_extractor.extract_chunk_metadata_batch(chunks)
        except:
            enriched_chunks = chunks
        _step("chunk_metadata", 45)

        # 4. Create vector index
        self.vector_store.create_index(enriched_chunks)
        _step("indexing", 55)

        # 5. Generate query variations
        try:
            variations = self.llm_analyzer.generate_query_variations(query)
        except:
            variations = [query]
        _step("retrieval", 65)

        # 6. Retrieve relevant chunks
        try:
            top_per_query = max(1, self.top_k_retrieval // len(variations))
            retrieved = self.vector_store.search_multiple_queries(
                queries=variations,
                top_k_per_query=top_per_query
            )
        except:
            retrieved = []
        _step("retrieval", 75)

        # 7. Extract arguments with LLM
        analysis_output = self.llm_analyzer.analyze_chunks(retrieved, query)
        if not analysis_output.extracted_points:
            return self._empty_response(doc_metadata, doc_id, query, enriched_chunks, start_time)
        _step("analysis", 85)

        # 8. Refinement
        original = [p.dict() for p in analysis_output.extracted_points]

        try:
            refined_raw = self.llm_analyzer.refine_extraction(original, query)
        except:
            refined_raw = original

        refined = []
        for o, r in zip(original, refined_raw):
            merged = o.copy()
            merged.update({
                "summary": r.get("summary", o["summary"]),
                "supporting_quote": r.get("supporting_quote", o["supporting_quote"]),
                "legal_concepts": r.get("legal_concepts", o["legal_concepts"]),
                "importance_score": r.get("importance_score", o["importance_score"])
            })
            refined.append(merged)

        _step("refinement", 92)

        # 9. Collect retrieval scores
        retrieval_scores = {}
        for ch in retrieved:
            cid = ch.get("metadata", {}).get("chunk_id")
            if cid:
                retrieval_scores[cid] = max(
                    retrieval_scores.get(cid, 0.0),
                    float(ch.get("score", 0.0))
                )

        # 10. Ranking & deduplication
        final_models = self.post_processor.process_and_rank(refined, retrieval_scores)
        _step("post_processing", 97)

        # 11. Build final display data
        key_points = []
        for p in final_models:
            data = p.dict()
            data["citation"] = format_page_citation(
                data.get("page_start"),
                data.get("page_end"),
                data.get("line_start"),
                data.get("line_end")
            )
            key_points.append(data)

        confidence = self._calculate_confidence(final_models, [c.get("score", 0) for c in retrieved])

        result = {
            "document_metadata": doc_metadata,
            "document_id": doc_id,
            "query": query,
            "total_chunks": len(enriched_chunks),
            "key_points": key_points,
            "confidence": confidence,
            "timings": {"elapsed_seconds": round(time.time() - start_time, 2)}
        }

        _step("done", 100)
        return result


    def _empty_response(self, doc_metadata, doc_id, query, chunks, start_time):
        return {
            "document_metadata": doc_metadata,
            "document_id": doc_id,
            "query": query,
            "total_chunks": len(chunks),
            "key_points": [],
            "confidence": 0.0,
            "timings": {"elapsed_seconds": round(time.time() - start_time, 2)}
        }

    def _calculate_confidence(self, points, retrieval_scores):
        if not points:
            return 0.0
        importance_avg = sum(p.importance_score for p in points) / len(points)
        retrieval_avg = sum(sorted(retrieval_scores, reverse=True)[:10]) / max(1, len(retrieval_scores))
        return round(min(1.0, max(0.0, importance_avg * 0.6 + retrieval_avg * 0.4)), 2)
