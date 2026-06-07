"""
app.py — Fake News Detection System
====================================
Streamlit web interface for real-time fake news classification.
Group: Antra, Prerina, Ain-Ul-Shuba | Sukkur IBA University | Spring 2026
"""

import streamlit as st
import pickle
import os
import time
from utils import preprocess_text

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Fake News Detector",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=DM+Mono:wght@400;500&display=swap');

html, body, [class*="css"] { font-family: 'Space Grotesk', sans-serif; }

.stApp { background: #0a0a0f; }

.main .block-container { padding-top: 2rem; padding-bottom: 3rem; max-width: 1100px; }

/* Hero */
.hero-wrap { text-align: center; padding: 2.5rem 1rem 1.5rem; }
.hero-badge {
    display: inline-block;
    background: rgba(99,102,241,0.15);
    border: 1px solid rgba(99,102,241,0.4);
    color: #818cf8;
    border-radius: 999px;
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    padding: 0.3rem 1rem;
    margin-bottom: 1rem;
}
.hero-title {
    font-size: 3.2rem;
    font-weight: 700;
    background: linear-gradient(135deg, #f8fafc 30%, #818cf8 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    line-height: 1.1;
    margin-bottom: 0.6rem;
}
.hero-sub {
    color: #64748b;
    font-size: 1rem;
    max-width: 500px;
    margin: 0 auto 1.5rem;
}

/* Cards */
.card {
    background: #12121a;
    border: 1px solid #1e1e2e;
    border-radius: 14px;
    padding: 1.4rem 1.5rem;
    margin-bottom: 1.2rem;
}
.card-label {
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: #475569;
    font-weight: 600;
    margin-bottom: 0.5rem;
}

/* Result Boxes */
.result-real {
    background: linear-gradient(135deg, rgba(16,185,129,0.12), rgba(5,150,105,0.08));
    border: 1.5px solid #10b981;
    border-radius: 14px;
    padding: 2rem;
    text-align: center;
    animation: fadeIn 0.4s ease;
}
.result-fake {
    background: linear-gradient(135deg, rgba(239,68,68,0.12), rgba(185,28,28,0.08));
    border: 1.5px solid #ef4444;
    border-radius: 14px;
    padding: 2rem;
    text-align: center;
    animation: fadeIn 0.4s ease;
}
.result-icon { font-size: 3rem; margin-bottom: 0.5rem; }
.result-label {
    font-size: 2rem;
    font-weight: 700;
    letter-spacing: 0.05em;
    margin-bottom: 0.3rem;
}
.result-real .result-label { color: #10b981; }
.result-fake .result-label { color: #ef4444; }
.result-conf { color: #94a3b8; font-size: 0.9rem; }

/* Probability bar */
.prob-bar-wrap { margin: 1rem 0; }
.prob-row { display: flex; align-items: center; gap: 0.7rem; margin-bottom: 0.5rem; }
.prob-label { color: #94a3b8; font-size: 0.82rem; width: 36px; }
.prob-track { flex: 1; height: 8px; background: #1e2030; border-radius: 99px; overflow: hidden; }
.prob-fill-real { height: 100%; background: linear-gradient(90deg,#059669,#10b981); border-radius: 99px; transition: width 0.6s ease; }
.prob-fill-fake { height: 100%; background: linear-gradient(90deg,#dc2626,#ef4444); border-radius: 99px; transition: width 0.6s ease; }
.prob-pct { color: #e2e8f0; font-family: 'DM Mono', monospace; font-size: 0.82rem; width: 44px; text-align: right; }

/* Stat chips */
.stat-row { display: flex; gap: 0.8rem; flex-wrap: wrap; margin-bottom: 1.2rem; }
.stat-chip {
    background: #12121a;
    border: 1px solid #1e1e2e;
    border-radius: 8px;
    padding: 0.5rem 0.9rem;
    text-align: center;
    flex: 1;
    min-width: 90px;
}
.stat-num { font-size: 1.4rem; font-weight: 700; color: #818cf8; font-family: 'DM Mono', monospace; }
.stat-lbl { color: #475569; font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.08em; }

/* How it works steps */
.step { display: flex; gap: 1rem; align-items: flex-start; margin-bottom: 1rem; }
.step-num {
    background: rgba(99,102,241,0.2);
    color: #818cf8;
    border-radius: 50%;
    width: 28px; height: 28px;
    display: flex; align-items: center; justify-content: center;
    font-size: 0.78rem; font-weight: 700;
    flex-shrink: 0;
}
.step-text { color: #94a3b8; font-size: 0.88rem; line-height: 1.5; padding-top: 4px; }
.step-text strong { color: #e2e8f0; }

/* Sidebar */
.sidebar-title { color: #e2e8f0; font-weight: 600; font-size: 0.9rem; margin-bottom: 0.4rem; }
.sidebar-val { color: #818cf8; font-family: 'DM Mono', monospace; font-size: 0.85rem; }

/* Warning / history */
.history-item {
    background: #0d0d14;
    border: 1px solid #1e1e2e;
    border-radius: 8px;
    padding: 0.7rem 1rem;
    margin-bottom: 0.6rem;
    font-size: 0.82rem;
    color: #94a3b8;
}
.tag-real { display:inline-block; background:rgba(16,185,129,0.15); color:#10b981; border-radius:4px; padding:1px 8px; font-size:0.75rem; font-weight:600; }
.tag-fake { display:inline-block; background:rgba(239,68,68,0.15); color:#ef4444; border-radius:4px; padding:1px 8px; font-size:0.75rem; font-weight:600; }

@keyframes fadeIn { from { opacity:0; transform:translateY(8px); } to { opacity:1; transform:translateY(0); } }

/* Button */
div.stButton > button {
    background: linear-gradient(135deg, #4f46e5, #818cf8);
    color: white; border: none; border-radius: 10px;
    padding: 0.65rem 2.2rem;
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 600; font-size: 1rem;
    width: 100%; cursor: pointer;
    transition: opacity 0.2s, transform 0.1s;
}
div.stButton > button:hover { opacity: 0.88; transform: translateY(-1px); }
div.stButton > button:active { transform: translateY(0); }

.stTextArea textarea {
    background: #0d0d14 !important;
    color: #e2e8f0 !important;
    border: 1px solid #1e2030 !important;
    border-radius: 10px !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-size: 0.92rem !important;
}
.stTextArea textarea:focus { border-color: #4f46e5 !important; }

hr { border-color: #1e1e2e !important; }

/* Sidebar styling */
section[data-testid="stSidebar"] {
    background: #0d0d14;
    border-right: 1px solid #1e1e2e;
}
</style>
""", unsafe_allow_html=True)


# ── Session State ──────────────────────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []


# ── Load Model ─────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    if not os.path.exists("model.pkl") or not os.path.exists("vectorizer.pkl"):
        return None, None
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    return model, vectorizer

model, vectorizer = load_model()


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🔍 Fake News Detector")
    st.markdown("<hr>", unsafe_allow_html=True)

    st.markdown("**Model Details**")
    st.markdown("""
    <div class='card' style='padding:1rem;'>
        <div class='card-label'>Algorithm</div>
        <div class='sidebar-val'>Logistic Regression</div>
        <div class='card-label' style='margin-top:0.8rem;'>Feature Extraction</div>
        <div class='sidebar-val'>TF-IDF (10,000 features)</div>
        <div class='card-label' style='margin-top:0.8rem;'>N-gram Range</div>
        <div class='sidebar-val'>Unigrams + Bigrams</div>
        <div class='card-label' style='margin-top:0.8rem;'>Expected Accuracy</div>
        <div class='sidebar-val'>~98%</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>**How It Works**", unsafe_allow_html=True)
    st.markdown("""
    <div class='step'>
        <div class='step-num'>1</div>
        <div class='step-text'><strong>Preprocessing</strong> — lowercase, remove URLs, punctuation & stopwords</div>
    </div>
    <div class='step'>
        <div class='step-num'>2</div>
        <div class='step-text'><strong>TF-IDF Vectorization</strong> — converts text into numerical feature vectors</div>
    </div>
    <div class='step'>
        <div class='step-num'>3</div>
        <div class='step-text'><strong>Classification</strong> — Logistic Regression outputs REAL or FAKE with a confidence score</div>
    </div>
    """, unsafe_allow_html=True)

    if st.session_state.history:
        st.markdown("<br>**Recent Analyses**", unsafe_allow_html=True)
        for item in reversed(st.session_state.history[-5:]):
            tag = "tag-real" if item["result"] == "REAL" else "tag-fake"
            label = "✅ REAL" if item["result"] == "REAL" else "❌ FAKE"
            st.markdown(f"""
            <div class='history-item'>
                <span class='{tag}'>{label}</span> &nbsp;
                <span style='color:#475569;'>{item['conf']}% confidence</span><br>
                <span style='color:#64748b;font-size:0.78rem;'>{item['preview']}</span>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("<p style='color:#334155;font-size:0.75rem;text-align:center;'>Antra · Prerina · Ain-Ul-Shuba<br>Sukkur IBA University · Spring 2026</p>", unsafe_allow_html=True)


# ── Main Content ───────────────────────────────────────────────────────────────
st.markdown("""
<div class='hero-wrap'>
    <div class='hero-badge'>AI & NLP · Spring 2026</div>
    <div class='hero-title'>Fake News<br>Detection System</div>
    <div class='hero-sub'>Paste any news article or headline and our ML model will classify it instantly.</div>
</div>
""", unsafe_allow_html=True)

# Model warning
if model is None:
    st.warning("⚠️ **Model not found.** Run `python train_model.py` first to train and save the model.")
    st.code("python train_model.py", language="bash")
    st.stop()

# Stats row
st.markdown("""
<div class='stat-row'>
    <div class='stat-chip'><div class='stat-num'>~98%</div><div class='stat-lbl'>LR Accuracy</div></div>
    <div class='stat-chip'><div class='stat-num'>~94%</div><div class='stat-lbl'>NB Accuracy</div></div>
    <div class='stat-chip'><div class='stat-num'>10K</div><div class='stat-lbl'>TF-IDF Features</div></div>
    <div class='stat-chip'><div class='stat-num'>44K+</div><div class='stat-lbl'>Training Articles</div></div>
</div>
""", unsafe_allow_html=True)

# ── Input Area ─────────────────────────────────────────────────────────────────
col_left, col_right = st.columns([3, 2], gap="large")

with col_left:
    st.markdown("<div class='card-label'>NEWS ARTICLE OR HEADLINE</div>", unsafe_allow_html=True)
    news_input = st.text_area(
        label="",
        height=220,
        placeholder="Paste a news article or headline here...\n\nExample: 'Scientists at MIT have developed a new AI system that can detect diseases with 99% accuracy...'"
    )

    word_count = len(news_input.split()) if news_input.strip() else 0
    char_count = len(news_input)
    st.markdown(f"<p style='color:#334155; font-size:0.78rem; margin-top:-0.5rem;'>{word_count} words · {char_count} characters</p>", unsafe_allow_html=True)

    analyze_btn = st.button("🔍 Analyze News Article")

with col_right:
    if analyze_btn:
        if not news_input.strip():
            st.error("Please enter some news text to analyze.")
        elif word_count < 5:
            st.warning("Please enter at least a few words for a reliable prediction.")
        else:
            with st.spinner("Analyzing..."):
                time.sleep(0.4)  # Small delay for UX
                cleaned = preprocess_text(news_input)
                features = vectorizer.transform([cleaned])
                prediction = model.predict(features)[0]
                proba = model.predict_proba(features)[0]
                real_pct = round(proba[1] * 100, 1)
                fake_pct = round(proba[0] * 100, 1)
                confidence = round(max(proba) * 100, 1)
                result_str = "REAL" if prediction == 1 else "FAKE"

            # Store in history
            st.session_state.history.append({
                "result": result_str,
                "conf": confidence,
                "preview": news_input[:60] + "..." if len(news_input) > 60 else news_input,
            })

            # Result card
            if prediction == 1:
                st.markdown(f"""
                <div class='result-real'>
                    <div class='result-icon'>✅</div>
                    <div class='result-label'>REAL NEWS</div>
                    <div class='result-conf'>Confidence: {confidence}%</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class='result-fake'>
                    <div class='result-icon'>❌</div>
                    <div class='result-label'>FAKE NEWS</div>
                    <div class='result-conf'>Confidence: {confidence}%</div>
                </div>
                """, unsafe_allow_html=True)

            # Probability bars
            st.markdown(f"""
            <div class='prob-bar-wrap' style='margin-top:1.2rem;'>
                <div class='card-label'>PROBABILITY BREAKDOWN</div>
                <div class='prob-row'>
                    <div class='prob-label' style='color:#10b981;'>Real</div>
                    <div class='prob-track'>
                        <div class='prob-fill-real' style='width:{real_pct}%;'></div>
                    </div>
                    <div class='prob-pct'>{real_pct}%</div>
                </div>
                <div class='prob-row'>
                    <div class='prob-label' style='color:#ef4444;'>Fake</div>
                    <div class='prob-track'>
                        <div class='prob-fill-fake' style='width:{fake_pct}%;'></div>
                    </div>
                    <div class='prob-pct'>{fake_pct}%</div>
                </div>
            </div>
            <div style='color:#475569; font-size:0.8rem; margin-top:0.8rem;'>
                Words analyzed: <strong style='color:#818cf8;'>{word_count}</strong>
                &nbsp;|&nbsp; Cleaned tokens: <strong style='color:#818cf8;'>{len(cleaned.split())}</strong>
            </div>
            """, unsafe_allow_html=True)

    else:
        # Placeholder state
        st.markdown("""
        <div style='text-align:center; padding: 3rem 1rem; color:#334155;'>
            <div style='font-size:3rem; margin-bottom:1rem;'>🔍</div>
            <div style='font-size:0.9rem;'>Your result will appear here after analysis.</div>
        </div>
        """, unsafe_allow_html=True)


# ── Example Articles ───────────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
with st.expander("💡 Try Example News Snippets"):
    ex_col1, ex_col2 = st.columns(2)
    with ex_col1:
        st.markdown("**Example — Likely Real:**")
        st.code(
            "The Federal Reserve raised interest rates by 25 basis points on Wednesday, "
            "citing continued concerns about inflation. The decision was unanimous among "
            "the 12-member Federal Open Market Committee.",
            language=None
        )
    with ex_col2:
        st.markdown("**Example — Likely Fake:**")
        st.code(
            "BREAKING: Scientists CONFIRM that drinking bleach every morning CURES all "
            "known diseases! Government hiding the truth!! Share before it gets deleted!!!",
            language=None
        )

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("""
<p style='text-align:center; color:#1e293b; font-size:0.8rem;'>
    Fake News Detection System · Antra, Prerina & Ain-Ul-Shuba · Sukkur IBA University · Spring 2026
</p>
""", unsafe_allow_html=True)
