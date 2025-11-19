import pdfplumber
from typing import List, Dict, Tuple
from loguru import logger
import hashlib


class PDFProcessor:
    def __init__(self, chunk_size: int = 1500, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def process_pdf(self, pdf_path: str) -> Tuple[List[Dict], Dict, str]:

        logger.info(f"Processing PDF: {pdf_path}")

        document_id = self._generate_doc_id(pdf_path)

        pages = self._extract_pages(pdf_path)
        lines = self._extract_lines_with_positions(pages)
        chunks = self._create_chunks(lines, document_id)

        metadata = {
            "total_pages": len(pages),
            "document_id": document_id
        }

        logger.info(f"Extracted {len(chunks)} chunks from {len(pages)} pages")
        return chunks, metadata, document_id

    def get_sample_text(self, pdf_path: str, num_pages: int = 3) -> str:
        """Returns first few pages for LLM-based metadata extraction."""
        try:
            with pdfplumber.open(pdf_path) as pdf:
                pages_to_sample = min(num_pages, len(pdf.pages))
                texts = []
                for i in range(pages_to_sample):
                    t = pdf.pages[i].extract_text()
                    if t:
                        texts.append(t)
                return "\n\n".join(texts)

        except Exception as e:
            logger.error(f"Error extracting sample text: {e}")
            return ""

    def _extract_pages(self, pdf_path: str) -> List[Dict]:
        """Extract all pages with plain text."""
        pages = []

        try:
            with pdfplumber.open(pdf_path) as pdf:
                for num, page in enumerate(pdf.pages, start=1):
                    text = page.extract_text() or ""
                    pages.append({"page_number": num, "text": text})

        except Exception as e:
            logger.error(f"Failed to load PDF: {e}")
            raise

        return pages

    def _extract_lines_with_positions(self, pages: List[Dict]) -> List[Dict]:
        """Split pages into lines and attach page + line numbers."""
        lines = []

        for page in pages:
            page_num = page["page_number"]
            raw_lines = page["text"].split("\n")

            for i, line in enumerate(raw_lines, start=1):
                line = line.strip()
                if line:
                    lines.append({
                        "page": page_num,
                        "line": i,
                        "text": line
                    })

        return lines

    def _create_chunks(self, lines: List[Dict], doc_id: str) -> List[Dict]:
        """Create overlapping chunks from line-level data."""
        chunks = []
        buffer = []
        length = 0
        index = 0

        i = 0
        while i < len(lines):
            ln = lines[i]
            buffer.append(ln)
            length += len(ln["text"])

            if length >= self.chunk_size:
                chunk = self._finalize_chunk(buffer, doc_id, index)
                chunks.append(chunk)
                index += 1

                # Create overlap buffer
                overlap = []
                overlap_chars = 0
                for j in range(len(buffer) - 1, -1, -1):
                    overlap.insert(0, buffer[j])
                    overlap_chars += len(buffer[j]["text"])
                    if overlap_chars >= self.chunk_overlap:
                        break

                buffer = overlap[:]
                length = overlap_chars

            i += 1

        # Add final tail chunk
        if buffer:
            chunk = self._finalize_chunk(buffer, doc_id, index)
            chunks.append(chunk)

        return chunks

    def _finalize_chunk(self, lines: List[Dict], doc_id: str, index: int) -> Dict:
        """Convert list of lines into a formatted chunk dict."""
        text = "\n".join(line["text"] for line in lines)

        return {
            "text": text,
            "metadata": {
                "document_id": doc_id,
                "chunk_id": f"{doc_id}_chunk_{index:04d}",
                "chunk_index": index,
                "page_start": lines[0]["page"],
                "page_end": lines[-1]["page"],
                "line_start": lines[0]["line"],
                "line_end": lines[-1]["line"],
                "char_count": len(text)
            }
        }

    def _generate_doc_id(self, pdf_path: str) -> str:
        """Creates deterministic document ID using MD5 hash."""
        return hashlib.md5(pdf_path.encode()).hexdigest()[:12]

