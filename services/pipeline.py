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


    def analyze(
        self,
        pdf_path: str,
        query: str,
        progress_callback: Optional[Callable[[str, int], None]] = None
    ) -> Dict[str, Any]:

        start_time = time.time()

        def _report(stage: str, pct: int):
            if progress_callback:
                try: progress_callback(stage, int(pct))
                except: pass

        logger.info(f"Starting analysis pipeline for {pdf_path}")
        _report("start", 1)

        # PDF → chunks
        _report("processing_pdf", 5)
        chunks, doc_metadata, doc_id = self.pdf_processor.process_pdf(pdf_path)
        _report("processing_pdf", 20)

        # Doc-level metadata
        sample = self.pdf_processor.get_sample_text(pdf_path)
        try:
            llm_meta = self.metadata_extractor.extract_document_metadata(sample)
            doc_metadata.update(llm_meta)
        except Exception as e:
            logger.warning(f"Document metadata failed: {e}")
        _report("doc_metadata", 30)

        # Chunk-level metadata 
        try:
            enriched_chunks = self.metadata_extractor.extract_chunk_metadata_batch(chunks)
        except Exception as e:
            logger.warning(f"Chunk metadata failed: {e}")
            enriched_chunks = chunks
        _report("chunk_metadata", 45)

        # Vector index 
        try:
            self.vector_store.create_index(enriched_chunks)
        except Exception as e:
            logger.error(f"Index creation failed: {e}")
            raise

        _report("indexing", 55)

        # Query variations
        try:
            query_variations = self.llm_analyzer.generate_query_variations(query)
        except:
            query_variations = [query]

        _report("retrieval", 65)

        # Retrieval across variations
        try:
            top_k_per_query = max(1, self.top_k_retrieval // len(query_variations))
            retrieved_chunks = self.vector_store.search_multiple_queries(
                queries=query_variations,
                top_k_per_query=top_k_per_query
            )
        except Exception as e:
            logger.error(f"Retrieval failed: {e}")
            retrieved_chunks = []

        _report("retrieval", 75)

        #LLM extraction 
        analysis_output = self.llm_analyzer.analyze_chunks(retrieved_chunks, query)
        if not analysis_output.extracted_points:
            logger.warning("LLM returned no extracted points.")
            return self._empty_response(doc_metadata, doc_id, query, enriched_chunks, start_time)

        _report("analysis", 85)

        #Refinement 
        original_points = [p.dict() for p in analysis_output.extracted_points]

        try:
            refined_raw = self.llm_analyzer.refine_extraction(original_points, query)
        except Exception as e:
            logger.error(f"Refinement failed: {e}")
            refined_raw = original_points

        refined_points = []
        for orig, ref in zip(original_points, refined_raw):
            merged = orig.copy()
            merged.update({
                "summary": ref.get("summary", orig.get("summary")),
                "supporting_quote": ref.get("supporting_quote", orig.get("supporting_quote")),
                "legal_concepts": ref.get("legal_concepts", orig.get("legal_concepts")),
                "importance_score": ref.get("importance_score", orig.get("importance_score"))
            })
            refined_points.append(merged)

        _report("refinement", 92)

        retrieval_scores = {}
        for ch in retrieved_chunks:
            cid = ch.get("metadata", {}).get("chunk_id")
            if cid:
                retrieval_scores[cid] = max(
                    retrieval_scores.get(cid, 0.0),
                    float(ch.get("score", 0.0))
                )

        try:
            final_models = self.post_processor.process_and_rank(refined_points, retrieval_scores)
        except Exception as e:
            logger.error(f"Post-processing failed: {e}")
            final_models = []

        _report("post_processing", 97)

        key_points = []
        for p in final_models:
            d = p.dict()
            d["citation"] = format_page_citation(
                d.get("page_start"), d.get("page_end"),
                d.get("line_start"), d.get("line_end")
            )
            key_points.append(d)

        # Confidence score
        all_scores = [c.get("score", 0.0) for c in retrieved_chunks]
        confidence = self._calculate_confidence(final_models, all_scores)

        result = {
            "document_metadata": doc_metadata,
            "document_id": doc_id,
            "query": query,
            "total_chunks": len(enriched_chunks),
            "key_points": key_points,
            "confidence": confidence,
            "timings": {"elapsed_seconds": round(time.time() - start_time, 2)}
        }

        _report("done", 100)
        return result


    def _empty_response(self, doc_metadata, doc_id, query, chunks, start):
        return {
            "document_metadata": doc_metadata,
            "document_id": doc_id,
            "query": query,
            "total_chunks": len(chunks),
            "key_points": [],
            "confidence": 0.0,
            "timings": {"elapsed_seconds": round(time.time() - start, 2)}
        }


    def _calculate_confidence(self, points, retrieval_scores):
        if not points:
            return 0.0

        importance_avg = sum([float(p.importance_score) for p in points]) / len(points)
        retrieval_avg = 0.0
        if retrieval_scores:
            top = sorted(retrieval_scores, reverse=True)[:10]
            retrieval_avg = sum(top) / len(top)

        conf = (importance_avg * 0.6) + (retrieval_avg * 0.4)
        return round(max(0.0, min(1.0, conf)), 2)
