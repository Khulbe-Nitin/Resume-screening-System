import re
from pathlib import Path
import fitz  # PyMuPDF
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

st.set_page_config(page_title="AI Resume Screener", layout="wide")

# --- 1. CONFIG & ONTOLOGY ---
SECTION_WEIGHTS = {
    "experience": 1.5,
    "projects": 1.5,
    "skills": 1.0,
    "education": 0.8,
    "other": 1.0,
}

SKILL_ONTOLOGY = {
    "python": ["python", "python3"],
    "llm_rag": ["rag", "retrieval augmented generation", "langchain", "llamaindex", "llm"],
    "vector_db": ["faiss", "chroma", "chromadb", "pinecone", "qdrant", "weaviate", "milvus"],
    "embeddings": ["sentence-transformers", "embeddings", "word2vec", "huggingface", "bert"],
    "backend_api": ["fastapi", "flask", "rest api", "django"],
    "evaluation": ["ragas", "ndcg", "precision", "recall", "f1", "evaluation"],
}

NEGATION_PATTERNS = [
    r"\bno\s+(?:prior\s+)?(?:experience|implementation|exposure|knowledge)\b",
    r"\bnot\s+(?:implemented|used|built|familiar|worked)\b",
    r"\blimited\s+(?:experience|exposure|knowledge)\b",
    r"\bwithout\s+(?:using|implementing)\b",
    r"\blacks?\s+experience\b",
]

WEAK_QUALIFIERS = [
    r"\btutorial[- ]based\b",
    r"\bbasic\s+understanding\b",
    r"\bbeginner\b",
    r"\btheoretical\s+only\b",
]


# --- 2. CACHED MODEL LOADER ---
@st.cache_resource
def load_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")


# --- 3. HELPER FUNCTIONS ---
def clean_text(text):
    text = re.sub(r"\(cid:\d+\)", "•", text)
    return re.sub(r"\s+", " ", text).strip()


def detect_section(line_text, is_heading):
    text_lower = line_text.lower().strip()
    if not is_heading and len(text_lower.split()) > 4:
        return None
    if any(k in text_lower for k in ["experience", "employment", "work history", "internship"]):
        return "experience"
    if any(k in text_lower for k in ["project", "personal build", "academic project"]):
        return "projects"
    if any(k in text_lower for k in ["skill", "technologies", "tech stack", "tools"]):
        return "skills"
    if any(k in text_lower for k in ["education", "academic", "qualification"]):
        return "education"
    return None


def check_polarity_multiplier(text):
    text_lower = text.lower()
    for pat in NEGATION_PATTERNS:
        if re.search(pat, text_lower):
            return 0.0
    for pat in WEAK_QUALIFIERS:
        if re.search(pat, text_lower):
            return 0.5
    return 1.0


def parse_pdf_stream(pdf_bytes, resume_name):
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    chunks = []
    current_section = "other"
    full_text_lines = []

    font_sizes = [
        s["size"]
        for page in doc
        for b in page.get_text("dict")["blocks"]
        for l in b.get("lines", [])
        for s in l.get("spans", [])
        if s["text"].strip()
    ]
    median_size = np.median(font_sizes) if font_sizes else 10.0

    for page in doc:
        for b in page.get_text("dict")["blocks"]:
            for l in b.get("lines", []):
                line_str = clean_text(" ".join(s["text"] for s in l.get("spans", [])))
                if not line_str or len(line_str) < 4:
                    continue

                max_size = max((s["size"] for s in l.get("spans", [])), default=median_size)
                is_bold = any("bold" in s["font"].lower() for s in l.get("spans", []))
                is_heading = (max_size > median_size * 1.08) or (is_bold and len(line_str.split()) <= 4)

                detected = detect_section(line_str, is_heading)
                if detected:
                    current_section = detected
                    continue

                full_text_lines.append(line_str)
                chunks.append({
                    "resume_id": resume_name,
                    "section": current_section,
                    "weight": SECTION_WEIGHTS[current_section],
                    "polarity_mult": check_polarity_multiplier(line_str),
                    "text": line_str,
                })
    # Fallback: if no chunks were added (e.g., short or unusual PDF layout), add full text
    if not chunks and full_text_lines:
        chunks.append({
            "resume_id": resume_name,
            "section": "other",
            "weight": 1.0,
            "polarity_mult": 1.0,
            "text": " ".join(full_text_lines)[:500],
        })
    return "\n".join(full_text_lines), chunks


def extract_jd_requirements(jd_text):
    lines = [clean_text(l) for l in jd_text.split("\n") if len(clean_text(l)) > 15]
    reqs = [l.lstrip("•-*1234567890. ") for l in lines if not l.endswith(":")]
    return reqs[:10]


def compute_ontology_skill_score(resume_text):
    text_lower = resume_text.lower()
    matched = []
    for group, aliases in SKILL_ONTOLOGY.items():
        if any(re.search(rf"\b{re.escape(a)}\b", text_lower) for a in aliases):
            matched.append(group)
    return len(matched) / len(SKILL_ONTOLOGY), matched


# --- 4. STREAMLIT UI ---
st.title("🎯 Hybrid AI Resume Screening System")
st.caption("Layout-Aware Parsing (PyMuPDF) • Negation Trap Detection • Semantic + Ontology Ranking")

# Auto-discover repo files if present
root = Path(".").resolve()
default_jd_files = list(root.rglob("job_description.txt"))
default_jd_text = default_jd_files[0].read_text(encoding="utf-8") if default_jd_files else ""
repo_pdfs = sorted([p for p in root.rglob("*.pdf") if ".venv" not in str(p)])

with st.sidebar:
    st.header("⚙️ Configuration")
    data_source = st.radio(
        "Resume Source:",
        ["Load from Repository", "Upload PDFs Manually"],
        index=0 if repo_pdfs else 1,
    )
    uploaded_files = []
    if data_source == "Upload PDFs Manually":
        uploaded_files = st.file_uploader("Upload Resume PDFs", type=["pdf"], accept_multiple_files=True)
    else:
        st.info(f"Found {len(repo_pdfs)} PDFs in repository.")

    st.subheader("Scoring Weights")
    sem_weight = st.slider("Semantic Requirement Weight", 0.0, 1.0, 0.65, 0.05)
    skill_weight = round(1.0 - sem_weight, 2)
    st.write(f"Ontology Skill Weight: **{skill_weight}**")

jd_input = st.text_area("📋 Job Description", value=default_jd_text, height=180)

if st.button("🚀 Run Screening Pipeline", type="primary"):
    # Gather PDFs
    pdf_inputs = []
    if data_source == "Load from Repository":
        pdf_inputs = [(p.stem, p.read_bytes()) for p in repo_pdfs]
    else:
        pdf_inputs = [(Path(f.name).stem, f.read()) for f in uploaded_files]

    if not jd_input.strip() or not pdf_inputs:
        st.error("Please provide both a Job Description and at least one PDF resume.")
        st.stop()

    with st.spinner("Extracting layout blocks, checking negations, and computing embeddings..."):
        resumes_raw = {}
        all_chunks = []
        for name, b_data in pdf_inputs:
            raw_t, c_list = parse_pdf_stream(b_data, name)
            if c_list:
                resumes_raw[name] = raw_t
                all_chunks.extend(c_list)
            else:
                st.warning(f"Skipped '{name}.pdf' (no readable text found).")

        chunks_df = pd.DataFrame(all_chunks)
        jd_reqs = extract_jd_requirements(jd_input)

        model = load_embedder()
        jd_vecs = model.encode(jd_reqs, normalize_embeddings=True)
        chunk_vecs = model.encode(chunks_df["text"].tolist(), normalize_embeddings=True)

        sim_matrix = np.dot(jd_vecs, chunk_vecs.T)
        eff_weights = (chunks_df["weight"] * chunks_df["polarity_mult"]).values
        weighted_sim = sim_matrix * eff_weights[np.newaxis, :]

        candidate_ids = sorted(resumes_raw.keys())
        req_heatmap = pd.DataFrame(index=candidate_ids, columns=[f"R{i+1}" for i in range(len(jd_reqs))], dtype=float)
        evidence_log = {cid: [] for cid in candidate_ids}

        for cid in candidate_ids:
            c_indices = np.where((chunks_df["resume_id"] == cid).values)[0]
            if len(c_indices) == 0:
                continue
            for r_idx, req_text in enumerate(jd_reqs):
                scores = weighted_sim[r_idx, c_indices]
                best_loc = np.argmax(scores)
                best_glob = c_indices[best_loc]
                best_val = float(scores[best_loc])
                req_heatmap.loc[cid, f"R{r_idx+1}"] = round(best_val, 3)
                evidence_log[cid].append({
                    "Requirement": f"R{r_idx+1}: {req_text}",
                    "Section": chunks_df.loc[best_glob, "section"],
                    "Polarity_Mult": chunks_df.loc[best_glob, "polarity_mult"],
                    "Matched_Resume_Line": chunks_df.loc[best_glob, "text"],
                    "Score": round(best_val, 3),
                })

        # Baseline TF-IDF for comparison
        tfidf = TfidfVectorizer(stop_words="english")
        t_mat = tfidf.fit_transform([jd_input] + [resumes_raw[c] for c in candidate_ids])
        base_scores = cosine_similarity(t_mat[0:1], t_mat[1:]).flatten()

        rows = []
        for i, cid in enumerate(candidate_ids):
            sem_s = req_heatmap.loc[cid].mean()
            sk_s, matched_sk = compute_ontology_skill_score(resumes_raw[cid])
            hyb_s = sem_weight * sem_s + skill_weight * sk_s
            top_ev = sorted(evidence_log[cid], key=lambda x: x["Score"], reverse=True)[0]["Matched_Resume_Line"]
            rows.append({
                "Candidate": cid,
                "Hybrid_Score": round(hyb_s, 4),
                "Semantic_Score": round(sem_s, 4),
                "Skill_Score": round(sk_s, 4),
                "Baseline_TFIDF": round(float(base_scores[i]), 4),
                "Matched_Domains": ", ".join(matched_sk),
                "Top_Verified_Evidence": top_ev,
            })

        ranking_df = pd.DataFrame(rows).sort_values("Hybrid_Score", ascending=False).reset_index(drop=True)
        ranking_df.index += 1

    # --- 5. DISPLAY TABS ---
    tab1, tab2, tab3, tab4 = st.tabs([
        "🏆 Ranked Candidates",
        "🔥 Requirement Heatmap",
        "🔍 Candidate Evidence Inspector",
        "🛡️ Trap & Negation Audit",
    ])

    with tab1:
        st.subheader("Final Candidate Ranking")
        st.dataframe(ranking_df, use_container_width=True)
        csv_bytes = ranking_df.to_csv(index=True).encode("utf-8")
        st.download_button("📥 Download Ranking CSV", data=csv_bytes, file_name="ranking_results.csv", mime="text/csv")

    with tab2:
        st.subheader("Candidate vs. JD Requirement Coverage")
        with st.expander("View Extracted JD Requirements Key"):
            for idx, r in enumerate(jd_reqs, 1):
                st.write(f"**R{idx}:** {r}")
        fig, ax = plt.subplots(figsize=(10, max(4, len(candidate_ids) * 0.4)))
        top_c = ranking_df["Candidate"].head(12).tolist()
        sns.heatmap(req_heatmap.loc[top_c], annot=True, cmap="YlGnBu", fmt=".2f", ax=ax)
        st.pyplot(fig)

    with tab3:
        st.subheader("Line-Level Grounded Evidence")
        selected_cand = st.selectbox("Select Candidate to Inspect:", ranking_df["Candidate"].tolist())
        st.table(pd.DataFrame(evidence_log[selected_cand]))

    with tab4:
        st.subheader("Caught Negations & Low-Proficiency Qualifiers")
        st.caption("Lines automatically penalized (0.0x for negations, 0.5x for tutorial/basic claims) so keyword-stuffed traps fail.")
        traps_df = chunks_df[chunks_df["polarity_mult"] < 1.0][["resume_id", "section", "polarity_mult", "text"]]
        st.dataframe(traps_df, use_container_width=True)