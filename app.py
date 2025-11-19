import streamlit as st
import streamlit.components.v1 as components
from pathlib import Path
import time
import traceback

from services.pipeline import AnalysisPipeline
from utils.helpers import sanitize_filename

# Streamlit Page Config
st.set_page_config(
    page_title="Legal Brief Analyzer",
    page_icon="⚖️",
    layout="wide",
)

# Custom CSS
st.markdown("""
<style>
.main-header {
    text-align: center;
    padding: 2rem 0;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    border-radius: 10px;
    margin-bottom: 2rem;
}
.point-card {
    border: 1px solid #e0e0e0;
    border-radius: 10px;
    padding: 1.5rem;
    margin-bottom: 1.5rem;
    box-shadow: 0 2px 4px rgba(0,0,0,0.08);
}
.rank-badge {
    display: inline-block;
    width: 40px; height: 40px;
    border-radius: 50%;
    background: #667eea;
    color: #fff;
    text-align: center;
    line-height: 40px;
    font-weight: bold;
    margin-right: 1rem;
}
.stance-badge {
    padding: 0.25rem 0.7rem;
    border-radius: 15px;
    font-size: 0.8rem;
    font-weight:600;
    margin-right:0.5rem;
}
.plaintiff { background:#e3f2fd; color:#1976d2; }
.defendant { background:#ffebee; color:#c62828; }
.amicus { background:#f3e5f5; color:#7b1fa2; }
.neutral { background:#f5f5f5; color:#616161; }

.quote-box {
    background:#f8f9fa;
    padding:1rem;
    border-left: 4px solid #667eea;
    font-style:italic;
    color:#555;
    margin:0.8rem 0;
}

/* small responsiveness */
@media (max-width: 600px) {
  .point-card { padding: 1rem; }
  .rank-badge { width:36px; height:36px; line-height:36px; }
}
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="main-header">
    <h1>⚖️ Legal Brief Analyzer</h1>
    <p>AI-powered extraction of key arguments with citations</p>
</div>
""", unsafe_allow_html=True)

# Sidebar Controls
with st.sidebar:
    st.header("⚙️ Settings")

    num_points = st.slider("Number of Key Points", 5, 20, 10)
    chunk_size = st.slider("Chunk Size", 500, 2500, 1500)
    top_k_retrieval = st.slider("Retrieval Depth", 20, 120, 60)

    st.markdown("---")
    st.info("""
    **How it works:**
    1. Upload a PDF  
    2. Extracts text + metadata  
    3. Embeds → retrieves → LLM analyzes  
    4. Outputs key arguments with citations  
    """)

# Input Area
col_left, col_right = st.columns([2, 1])

with col_left:
    uploaded_file = st.file_uploader("📄 Upload a Legal Brief (PDF)", type=["pdf"])

with col_right:
    query = st.text_input("🔍 Search Query", value="key legal arguments for and against")

# Progress Display
progress_bar = st.progress(0)
progress_text = st.empty()

def progress_callback(stage: str, percent: int):
    progress_bar.progress(percent)
    progress_text.text(f"Stage: {stage} ({percent}%)")

# Normalizing stance for final display
def normalize_stance(raw):
    """Accepts enum, string, or broken value → returns final stance."""
    if raw is None:
        return "neutral"

    # If enum
    try:
        if hasattr(raw, "value"):
            raw = raw.value
    except:
        pass

    val = str(raw).strip().lower()

    # Fix weird values like "stance.for"
    if "." in val:
        val = val.split(".")[-1]

    mapping = {
        "for": "for",
        "plaintiff": "plaintiff",
        "defendant": "defendant",
        "against": "against",
        "amicus": "amicus",
        "neutral": "neutral",
        "unknown": "neutral"
    }

    return mapping.get(val, "neutral")

# Card Renderer
def render_card(point: dict):
    stance_key = normalize_stance(point.get("stance"))
    stance_display = stance_key.upper()

    stance_color = {
        "plaintiff": "plaintiff",
        "for": "plaintiff",
        "defendant": "defendant",
        "against": "defendant",
        "amicus": "amicus",
        "neutral": "neutral"
    }.get(stance_key, "neutral")

    # concepts
    concepts = point.get("legal_concepts", []) or []
    concept_badges = "".join(
        f"<span class='stance-badge' style='background:#e8eaf6;color:#3f51b5'>{c}</span>"
        for c in concepts
    )

    score_pct = int(round(point.get("importance_score", 0.0) * 100))
    summary = point.get("summary", "")
    quote = point.get("supporting_quote", "")
    citation = point.get("citation", "No citation")
    rank = point.get("final_rank", "?")

    html = f"""
    <div class='point-card'>
      <div style='display:flex; align-items:flex-start;'>
        <div class='rank-badge'>{rank}</div>
        <div style='flex:1;'>
          <div style='margin-bottom:0.6rem'>
            <span class='stance-badge {stance_color}'>{stance_display}</span>
            <span class='stance-badge' style='background:#eef;color:#114;'>Score: {score_pct}%</span>
          </div>

          <p style='font-size:1.05rem;color:#222'>{summary}</p>

          <div class='quote-box'>" {quote} "</div>

          <p><strong>📍 Citation:</strong> {citation}</p>

          <p style='margin-top:0.5rem'>{concept_badges}</p>
        </div>
      </div>
    </div>
    """

    components.html(html, height=260, scrolling=True)

# Main Action
if uploaded_file:
    if st.button("🚀 Analyze Document", type="primary", use_container_width=True):
        temp_dir = Path("temp_uploads")
        temp_dir.mkdir(exist_ok=True)

        safe_name = sanitize_filename(uploaded_file.name)
        file_path = temp_dir / safe_name
        file_path.write_bytes(uploaded_file.getbuffer())

        try:
            pipeline = AnalysisPipeline(
                final_k=num_points,
                chunk_size=chunk_size,
                top_k_retrieval=top_k_retrieval
            )

            progress_callback("starting", 1)
            t0 = time.time()

            # Call Pipeline
            try:
                result = pipeline.analyze(
                    pdf_path=str(file_path),
                    query=query,
                    progress_callback=progress_callback
                )
            except TypeError:
                # If pipeline doesn't support progress callback
                result = pipeline.analyze(pdf_path=str(file_path), query=query)

            elapsed = time.time() - t0
            progress_callback("done", 100)
            time.sleep(0.25)

            # Cleanup
            file_path.unlink(missing_ok=True)

            st.success(f"Analysis completed in {elapsed:.1f}s")

            meta = result.get("document_metadata", {})
            key_points = result.get("key_points", [])

            # Metrics
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Pages", meta.get("total_pages", "N/A"))
            c2.metric("Key Points", len(key_points))
            c3.metric("Confidence", f"{int(result.get('confidence', 0)*100)}%")
            c4.metric("Processing Time", f"{elapsed:.1f}s")

            st.markdown("---")

            # Document Info
            if meta.get("case_name"):
                st.subheader("📋 Document Details")
                m1, m2, m3 = st.columns(3)
                m1.write(f"**Case Name:** {meta.get('case_name')}")
                m2.write(f"**Type:** {meta.get('document_type', 'N/A')}")
                m3.write(f"**Court:** {meta.get('court', 'N/A')}")
                st.markdown("---")

            # Key Points Section
            st.subheader("🔑 Key Legal Arguments")
            if not key_points:
                st.info("No key points extracted.")

            for p in key_points:
                render_card(p)

        except Exception as e:
            st.error("❌ Analysis failed.")
            st.code(traceback.format_exc())

else:
    st.info("👆 Upload a PDF to begin")
