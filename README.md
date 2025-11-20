# ⚖️ Legal Brief Analyzer  
AI-powered tool that extracts the **top legal arguments** (for/against), key quotes, legal concepts, and citations (page/line) from any legal brief.

This project automates a lawyer’s preparation workflow by compressing large PDFs into the **10 most important legal points** with accurate references.

---

## 🚀 Features
- Upload any legal PDF  
- Extract key arguments (For / Against / Neutral)  
- Supporting quotes + page & line citations  
- Legal concepts & argument categories  
- Document metadata (case name, court, type)  
- Semantic retrieval using FAISS  
- LLM-based extraction + refinement  
- Duplicate removal + ranking  
- Clean Streamlit UI

---

## 🧠 Architecture Overview

### 1️⃣ PDF → Chunks  
Extract text page-wise → split into fixed-size chunks with:
- page number  
- line numbers  
- chunk_id  

### 2️⃣ Metadata Extraction  
LLM extracts:
- **Document-level metadata** → case name, document type, court  
- **Chunk-level metadata** → stance, importance_score, legal concepts, argument type  

### 3️⃣ FAISS Vector Store  
All chunks → embedding → FAISS index for fast semantic search.

### 4️⃣ Retrieval  
User query → embedding → retrieve most relevant chunks via semantic similarity.

### 5️⃣ Argument Extraction (LLM)  
LLM pulls arguments with:
- summary  
- stance  
- supporting quote  
- importance_score  
- legal concepts  
- citation placeholders  

### 6️⃣ Refinement  
LLM cleans summaries and improves quotes/legal concepts (scores unchanged).

### 7️⃣ Post-Processing  
- merge retrieval + importance scores  
- remove duplicate points (RapidFuzz)  
- rank top K  
- add final citations (page/line)  

---

## 🏗️ Tech Stack
- **Python**  
- **Streamlit** for UI  
- **FAISS** for vector search  
- **Groq LLaMA 3.3** for metadata + argument extraction  
- **RapidFuzz** for deduplication  
- **PDFPlumber** for text extraction  

---

## ▶️ Run Locally
```bash
pip install -r requirements.txt
streamlit run app.py
