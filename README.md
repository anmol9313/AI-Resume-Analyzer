import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import PyPDF2
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# --- 1. NLTK Resource Initialization ---
@st.cache_resource
def load_nltk_resources():
    nltk.download("punkt", quiet=True)
    nltk.download("stopwords", quiet=True)

load_nltk_resources()

# --- 2. Premium Page Layout Setup ---
st.set_page_config(
    page_title="TalentAlign Match Engine",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Unified SaaS CSS Layout (Fixes the nested white boxes)
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');
    
    /* Core Layout Font Enforcement */
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif !important;
    }
    
    /* Absolute visibility anchors for plain text descriptions */
    .saas-heading {
        color: #0F172A !important;
        font-weight: 700 !important;
        margin-bottom: 4px !important;
        font-size: 16px !important;
    }
    .saas-subtext {
        color: #475569 !important;
        font-size: 13px !important;
        font-weight: 400 !important;
        margin-bottom: 12px !important;
        line-height: 1.5 !important;
    }

    /* Target Streamlit's native file uploader and text area containers directly */
    div[data-testid="stFileUploader"], div[data-testid="stTextArea"] {
        background-color: #FFFFFF !important;
        border: 1px solid #E2E8F0 !important;
        border-radius: 14px !important;
        padding: 16px !important;
        box-shadow: 0 4px 6px -1px rgba(15, 23, 42, 0.03) !important;
        transition: all 0.2s ease-in-out !important;
    }
    
    /* Add clean interactive hover states directly onto the native inputs */
    div[data-testid="stFileUploader"]:hover, div[data-testid="stTextArea"]:hover {
        border-color: #CBD5E1 !important;
        box-shadow: 0 8px 12px -3px rgba(15, 23, 42, 0.06) !important;
    }
    
    /* Premium Title Badge */
    .badge {
        background: linear-gradient(135deg, #EEF2FF 0%, #E0E7FF 100%) !important;
        color: #4F46E5 !important;
        padding: 6px 14px;
        border-radius: 9999px;
        font-size: 11px;
        font-weight: 700;
        display: inline-block;
        margin-bottom: 16px;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    /* Metric Result Block */
    .metric-container {
        text-align: center;
        background-color: #FFFFFF !important;
        border: 1px solid #E2E8F0;
        border-radius: 20px;
        padding: 35px;
        box-shadow: 0 4px 10px rgba(15, 23, 42, 0.02);
    }
    .metric-value {
        font-size: 64px;
        font-weight: 800;
        background: linear-gradient(135deg, #2563EB 0%, #1D4ED8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        line-height: 1;
        margin-bottom: 12px;
        letter-spacing: -0.02em;
    }
    .metric-label {
        font-size: 13px;
        color: #64748B !important;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }

    /* Interactive Button Mechanics */
    div.stButton > button:first-child {
        background: linear-gradient(135deg, #1E293B 0%, #0F172A 100%) !important;
        color: #FFFFFF !important;
        border: none !important;
        padding: 14px 28px !important;
        border-radius: 12px !important;
        font-weight: 600 !important;
        font-size: 16px !important;
        box-shadow: 0 4px 12px rgba(15, 23, 42, 0.12) !important;
        transition: all 0.2s ease-in-out !important;
        width: 100%;
    }
    div.stButton > button:first-child:hover {
        background: linear-gradient(135deg, #0F172A 0%, #020617 100%) !important;
        transform: translateY(-1px) !important;
        box-shadow: 0 6px 16px rgba(15, 23, 42, 0.2) !important;
    }

    /* Flow Footer Framework */
    .saas-footer {
        width: 100%;
        color: #64748B !important;
        text-align: center;
        padding: 24px 10px 10px 10px;
        font-size: 12px;
        font-weight: 600;
        border-top: 1px solid #E2E8F0;
        letter-spacing: 0.05em;
        margin-top: 60px;
    }
    </style>
""", unsafe_allow_html=True)


# --- 3. Text Parsing & Analytics Methods ---
def extract_text_from_pdf(uploaded_file):
    try:
        pdf_reader = PyPDF2.PdfReader(uploaded_file)
        text = ""
        for page in pdf_reader.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted
        return text
    except Exception as e:
        st.error(f"Execution Error: Could not extract structural data layers from file: {e}")
        return ""

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def remove_stopwords(text):
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(text)
    return " ".join([word for word in words if word not in stop_words])
  
def calculate_similarity(resume_text, job_description):
    resume_processed = remove_stopwords(clean_text(resume_text))
    job_processed = remove_stopwords(clean_text(job_description))
    
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([resume_processed, job_processed])
    score = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0] * 100
    return round(score, 2), resume_processed, job_processed


# --- 4. Sidebar Panel Navigation ---
with st.sidebar:
    st.markdown("<div style='padding-top: 15px;'></div>", unsafe_allow_html=True)
    st.markdown("### 🧠 Diagnostic Hub")
    st.markdown("Automated alignment matching metrics using TF-IDF token frequency matching arrays.")
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    with st.expander("🛠️ Operational Matrix", expanded=True):
        st.markdown("""
        - **Proximity Weighting** Calculates vector semantic distance indices.
        - **Key Vocabulary Parsing** Extracts contextual requirements.
        - **Delta Tracking** Identifies target keyword deficiencies.
        
        """)


# --- 5. Main Application Logic ---

def main():
    # Structural Title Elements
    st.markdown("<span class='badge'>AI Resume Analyzer</span>", unsafe_allow_html=True)
    st.markdown("<h1 style='margin-top:0px; font-size:34px; font-weight:800; letter-spacing:-0.02em;'>Resume Compatibility Matrix</h1>", unsafe_allow_html=True)
    st.markdown("<p style='font-size:15px; color:#475569; margin-bottom: 25px;'>Parse profile summaries against enterprise target requisitions dynamically with live compliance metrics.</p>", unsafe_allow_html=True)

    # Input Matrix Layout split cleanly 
    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.markdown("<div class='saas-heading'>📄 Candidate Profile Summary</div>", unsafe_allow_html=True)
        st.markdown("<div class='saas-subtext'>Drop or browse an applicant profile summary structured in standard PDF layout format.</div>", unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Upload Profile PDF", type=['pdf'], label_visibility="collapsed")
        
    with col2:
        st.markdown("<div class='saas-heading'>📝 Requisition Benchmarks</div>", unsafe_allow_html=True)
        st.markdown("<div class='saas-subtext'>Provide objective core targets, role responsibilities, or direct job definition texts.</div>", unsafe_allow_html=True)
        job_description = st.text_area("Job Spec Inputs", height=105, label_visibility="collapsed", placeholder="Paste benchmark parameters or job specifications here...")

    st.markdown("<div style='margin-top: 25px;'></div>", unsafe_allow_html=True)

    # Processing Button
    if st.button("Execute Match Alignment"):
        if not uploaded_file:
            st.toast("⚠️ Context Missing: Please upload a primary candidate resume asset.", icon="📄")
            return
        if not job_description:
            st.toast("⚠️ Context Missing: Please define target job specifications.", icon="📝")
            return
        
        with st.spinner("Extracting parameters and processing language algorithms..."):
            resume_text = extract_text_from_pdf(uploaded_file)
            if not resume_text.strip():
                st.error("Extraction Terminated: The document source contains no active readable text structures.")
                return 
            
            similarity_score, resume_processed, job_processed = calculate_similarity(resume_text, job_description)

            # Results View Rendering
            st.markdown("<hr>", unsafe_allow_html=True)
            out_col1, out_col2 = st.columns([1, 2], gap="large")
            
            with out_col1:
                st.markdown(f"""
                <div class="metric-container">
                    <div class="metric-value">{similarity_score:.1f}%</div>
                    <div class="metric-label">Match Alignment Score</div>
                </div>
                """, unsafe_allow_html=True)
            
            with out_col2:
                st.markdown("<p style='font-weight:700; font-size:15px; margin-bottom:10px; color:#0F172A;'>Compliance Threshold Map</p>", unsafe_allow_html=True)
                st.progress(int(similarity_score))
                
                st.markdown("<div style='margin-top: 24px;'></div>", unsafe_allow_html=True)
                
                if similarity_score < 40:
                    st.markdown("""
                    <div style='background-color:#FEE2E2; border-left:5px solid #EF4444; padding:20px; border-radius:12px;'>
                        <span style='color:#991B1B; font-weight:800; font-size:14px; text-transform:uppercase;'>🚨 Low Compatibility Detected</span><br>
                        <span style='color:#7F1D1D; font-size:13.5px; font-weight:500; line-height:1.5;'>The scanned profile has low keyword convergence with target targets. Consider tailoring language strings closer to core requirements.</span>
                    </div>
                    """, unsafe_allow_html=True)
                elif similarity_score < 70:
                    st.markdown("""
                    <div style='background-color:#DBEAFE; border-left:5px solid #3B82F6; padding:20px; border-radius:12px;'>
                        <span style='color:#1E40AF; font-weight:800; font-size:14px; text-transform:uppercase;'>📈 Viable Threshold Achieved</span><br>
                        <span style='color:#1E3A8A; font-size:13.5px; font-weight:500; line-height:1.5;'>The profile covers fundamental criteria effectively. Integrating specific missing operational action verbs can optimize total ranking score.</span>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div style='background-color:#DCFCE7; border-left:5px solid #22C55E; padding:20px; border-radius:12px;'>
                        <span style='color:#166534; font-weight:800; font-size:14px; text-transform:uppercase;'>✅ High-Density Baseline Alignment</span><br>
                        <span style='color:#14532D; font-size:13.5px; font-weight:500; line-height:1.5;'>Strong organic match metrics observed. Candidate summary data indexes highly against specified requirements natively.</span>
                    </div>
                    """, unsafe_allow_html=True)

    # Static Non-blocking Document Flow Footer
    st.markdown("""
        
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
