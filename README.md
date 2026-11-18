# Resume-screening-System

A hybrid, explainable resume ranking tool. It reads PDF resumes, compares them with a job description and produces a ranked shortlist in which every score can be traced back to a specific line of the resume.

**Live demo:** https://resume-screening-system-fsjizphnzkmlmgjc86ycd6.streamlit.app/

Built as a B.Tech (CSE) project at Guru Tegh Bahadur 4th Centenary Engineering College (GGSIPU), batch 2023-27.

---

## Why this project

Keyword-based applicant tracking systems reward keyword stuffing, miss synonyms ("RAG" vs "retrieval-augmented generation") and cannot read negation ("no prior experience with FAISS" still counts as a FAISS match). This project combines layout-aware parsing, semantic matching and simple linguistic rules to avoid those mistakes while staying transparent to the recruiter.

## Features

- **Layout-aware PDF parsing (PyMuPDF).** Font size and boldness are used to detect section headings (experience, projects, skills, education). Each line becomes a chunk tagged with its section.
- **Section weighting.** Experience and projects count 1.5, skills 1.0, education 0.8, other 1.0, so demonstrated work counts more than a bare skill list.
- **Negation and weak-claim detection.** Lines such as "no prior experience" get a multiplier of 0.0; "tutorial-based", "beginner" or "basic understanding" get 0.5.
- **Semantic matching.** Job requirements and resume lines are embedded with `all-MiniLM-L6-v2` (Sentence-Transformers) and compared by cosine similarity. The best weighted line per requirement is kept as evidence.
- **Skill ontology score.** Six skill groups (Python, LLM/RAG, vector databases, embeddings, backend APIs, evaluation) with alias lists, so equivalent tools get fair credit.
- **Hybrid score.** `Hybrid = w x Semantic + (1 - w) x Skill`, with `w` adjustable on a sidebar slider (default 0.65).
- **TF-IDF baseline** shown next to the hybrid score for comparison.
- **Four result views:** ranked candidates (with CSV download), requirement heatmap, line-level evidence inspector, and a trap and negation audit.

## How it works

```
PDF resumes ──> PyMuPDF parser ──> section + weight + polarity per line ─┐
                                                                         ├─> weighted cosine matrix ──> semantic score ─┐
Job description ──> requirement extractor ──> sentence embeddings ───────┘                                              ├─> hybrid score ──> ranking
                                                  resume text ──> skill ontology ──> skill score ───────────────────────┘
                                                  resume text ──> TF-IDF vs job description ──> baseline (comparison only)
```

For a candidate `c` and requirement `i`:

```
score(c, i)  = max over lines j of c of  cosine(req_i, line_j) x section_weight_j x polarity_j
Semantic(c)  = mean over i of score(c, i)
Skill(c)     = matched skill groups / 6
```

## Project structure

```
.
├── app.py                      # Complete application (parsing, scoring, Streamlit UI)
├── requirements.txt            # Python dependencies
├── job_description.txt         # Sample job description (Generative AI / LLM Engineer)
├── resume_screener/
│   └── resumes/                # 20 synthetic sample resumes (PDF)
└── README.md
```

## Getting started

Requires Python 3.9 or later.

```bash
# 1. Clone
git clone https://github.com/Khulbe-Nitin/Resume-screening-System.git
cd Resume-screening-System

# 2. (Optional) virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run
streamlit run app.py
```

The first run downloads the `all-MiniLM-L6-v2` model (about 90 MB) from the Hugging Face Hub. The app opens at http://localhost:8501.

## Usage

1. In the sidebar, choose **Load from Repository** (uses the 20 sample resumes) or **Upload PDFs Manually**.
2. Review or edit the job description.
3. Adjust the **Semantic Requirement Weight** slider if you want more weight on skill coverage (lower) or on semantic evidence (higher).
4. Click **Run Screening Pipeline**.
5. Explore the four tabs and download the ranking as CSV.

Note: the results shown belong to the settings used when the button was last pressed, so click the button again after moving the slider.

## Sample results

Top ten of the 20 sample resumes at the default weights (0.65 semantic, 0.35 skill), from the deployed app:

| Rank | Candidate | Hybrid | Semantic | Skill | TF-IDF |
|-----:|-----------|-------:|---------:|------:|-------:|
| 1 | resume_20_ananya_iyer | 0.6943 | 0.6195 | 0.8333 | 0.1409 |
| 2 | resume_01_aarav_mehta | 0.6774 | 0.5934 | 0.8333 | 0.1457 |
| 3 | resume_02_diya_nair | 0.6449 | 0.4537 | 1.0000 | 0.1178 |
| 4 | resume_07_aditya_bose | 0.6362 | 0.5301 | 0.8333 | 0.0986 |
| 5 | resume_04_ishita_rao | 0.6353 | 0.5286 | 0.8333 | 0.1178 |
| 6 | resume_08_sana_qureshi | 0.6296 | 0.5199 | 0.8333 | 0.1511 |
| 7 | resume_06_meera_shah | 0.6268 | 0.5156 | 0.8333 | 0.1178 |
| 8 | resume_05_rohan_kulkarni | 0.5872 | 0.5444 | 0.6667 | 0.1419 |
| 9 | resume_10_nisha_thomas | 0.5776 | 0.5296 | 0.6667 | 0.1263 |
| 10 | resume_09_vikram_desai | 0.5479 | 0.5737 | 0.5000 | 0.1750 |

The TF-IDF ranking and the skill-coverage ranking agree only moderately (Spearman rank correlation about 0.59), which is the reason for combining several signals. The negation audit flags two lines in the sample set (`resume_11_yash_malhotra` and `resume_20_ananya_iyer`).

## Known limitations

- Only text-based PDFs are supported (scanned resumes need OCR).
- The requirement extractor takes the first ten lines of the job description, so header lines (role, location) can take requirement slots.
- Negation and weak-claim detection uses a fixed list of regular expressions and acts on whole lines. For example, "no vector retrieval implementation" is not caught by the negation patterns.
- Wrapped lines in a PDF become separate chunks, and headings without a known keyword (such as "ADDITIONAL INFORMATION") do not change the current section.
- The skill ontology is hand-made for Generative AI roles and misses spelling variants such as "Hugging Face" (written as two words).
- Validated only on 20 synthetic resumes, not against real recruiter decisions.

## Roadmap

- Extract requirements only from the Responsibilities and Qualifications sections.
- Merge wrapped lines and split lines at semicolons before applying polarity rules.
- Extend the ontology with more aliases or a standard taxonomy.
- OCR support and DOCX resumes.
- Years-of-experience extraction.
- Fairness audit (rankings should not change when names or institutions change).
- Pinned dependency versions, automated tests and a Docker image.

## Responsible use

The system uses only the text content of the resume lines and no personal attributes. It is meant to help a recruiter decide what to read first, not to reject candidates automatically. The sample resumes are synthetic; for real resumes, host the app privately.

## Built with

Python, Streamlit, PyMuPDF, Sentence-Transformers, scikit-learn, NumPy, Pandas, Matplotlib, Seaborn.

## Author

Nitin Khulbe, B.Tech CSE, Guru Tegh Bahadur 4th Centenary Engineering College, New Delhi. Supervisor: Dr. Pardeep Singh (Dean Academics & HOD, CSE).
