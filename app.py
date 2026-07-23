import streamlit as st
from recommendation import CourseRecommender

# -------------------------------------------------
# Page Configuration
# -------------------------------------------------
st.set_page_config(
    page_title="AI Course Recommendation System",
    page_icon="🤖",
    layout="wide"
)

# -------------------------------------------------
# Custom CSS — Futuristic "AI Background" Theme
# -------------------------------------------------
st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@500;700;900&family=Rajdhani:wght@400;500;600&display=swap');

/* Animated neural-network style gradient background */
.stApp {
    background: radial-gradient(circle at 20% 20%, #0f2447 0%, #050b18 45%, #01040a 100%);
    background-attachment: fixed;
    color: #e6f1ff;
}

/* Subtle animated grid overlay to evoke a neural / circuit backdrop */
.stApp::before {
    content: "";
    position: fixed;
    inset: 0;
    background-image:
        linear-gradient(rgba(0, 229, 255, 0.06) 1px, transparent 1px),
        linear-gradient(90deg, rgba(0, 229, 255, 0.06) 1px, transparent 1px);
    background-size: 42px 42px;
    animation: driftGrid 30s linear infinite;
    pointer-events: none;
    z-index: 0;
}

@keyframes driftGrid {
    0% { background-position: 0 0; }
    100% { background-position: 420px 420px; }
}

.main-title{
    font-family: 'Orbitron', sans-serif;
    font-size: 42px;
    font-weight: 900;
    text-align: center;
    background: linear-gradient(90deg, #00e5ff, #7b61ff, #00e5ff);
    background-size: 200% auto;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    animation: shimmer 6s linear infinite;
    text-shadow: 0 0 25px rgba(0, 229, 255, 0.25);
    margin-bottom: 0;
}

@keyframes shimmer {
    0% { background-position: 0% center; }
    100% { background-position: 200% center; }
}

.sub-title{
    font-family: 'Rajdhani', sans-serif;
    font-size: 18px;
    text-align:center;
    color:#8fb8e0;
    letter-spacing: 1px;
    margin-top: 4px;
}

.course-card{
    font-family: 'Rajdhani', sans-serif;
    padding: 20px 22px;
    border-radius: 14px;
    background: linear-gradient(145deg, rgba(20, 35, 65, 0.75), rgba(10, 18, 38, 0.85));
    margin-bottom: 16px;
    border: 1px solid rgba(0, 229, 255, 0.25);
    border-left: 5px solid #00e5ff;
    box-shadow: 0 0 18px rgba(0, 229, 255, 0.08);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}

.course-card:hover{
    transform: translateY(-3px);
    box-shadow: 0 0 26px rgba(0, 229, 255, 0.25);
}

.course-card h4{
    color: #00e5ff;
    margin-bottom: 6px;
    font-family: 'Orbitron', sans-serif;
    font-size: 17px;
}

.course-card b {
    color: #7b61ff;
}

/* Sidebar styling */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #050b18 0%, #0a1830 100%);
    border-right: 1px solid rgba(0, 229, 255, 0.15);
}

/* Buttons */
div.stButton > button {
    font-family: 'Orbitron', sans-serif;
    background: linear-gradient(90deg, #00b8d4, #7b61ff);
    color: white;
    border: none;
    border-radius: 10px;
    padding: 10px 0;
    font-weight: 700;
    letter-spacing: 1px;
    box-shadow: 0 0 16px rgba(123, 97, 255, 0.35);
    transition: box-shadow 0.2s ease, transform 0.2s ease;
}

div.stButton > button:hover {
    box-shadow: 0 0 26px rgba(0, 229, 255, 0.55);
    transform: translateY(-1px);
    color: white;
}

/* Keep default text readable against dark background */
p, span, label, .stMarkdown, .stCaption {
    color: #cfe3f7;
}

</style>
""", unsafe_allow_html=True)

# -------------------------------------------------
# Load Recommendation Model
# -------------------------------------------------
@st.cache_resource
def load_model():
    return CourseRecommender()

recommender = load_model()

# -------------------------------------------------
# Header
# -------------------------------------------------
st.markdown(
    '<p class="main-title">🤖 AI Course Recommendation System</p>',
    unsafe_allow_html=True
)

st.markdown(
    '<p class="sub-title">Neural-powered course discovery using TF-IDF &amp; Cosine Similarity</p>',
    unsafe_allow_html=True
)

st.divider()

# -------------------------------------------------
# Sidebar
# -------------------------------------------------
st.sidebar.title("📡 About")

st.sidebar.info("""
This system recommends similar courses using:

✅ TF-IDF Vectorization

✅ Cosine Similarity

Built with:

• Python

• Streamlit

• Scikit-Learn

• Pandas
""")

st.sidebar.markdown("---")
st.sidebar.caption("🧠 Powered by a lightweight on-device recommendation model — no external API required.")

# -------------------------------------------------
# Course Selection
# -------------------------------------------------
course_list = recommender.get_course_list()

selected_course = st.selectbox(
    "🎯 Select a Course",
    course_list
)

top_n = st.slider(
    "🔢 Number of Recommendations",
    min_value=3,
    max_value=10,
    value=5
)

# -------------------------------------------------
# Recommendation Button
# -------------------------------------------------
if st.button("🚀 Recommend Courses", use_container_width=True):

    recommendations = recommender.recommend(
        selected_course,
        top_n
    )

    if not recommendations:
        st.warning("No recommendations found for the selected course.")
    else:
        st.success(
            f"Top {len(recommendations)} recommendations for '{selected_course}'"
        )

        st.write("")

        for i, course in enumerate(recommendations, start=1):
            st.markdown(
                f"""
<div class="course-card">
<h4>{i}. {course['Course_Name']}</h4>
<b>Category:</b> {course['Category']}&nbsp;&nbsp;|&nbsp;&nbsp;
<b>Level:</b> {course['Level']}&nbsp;&nbsp;|&nbsp;&nbsp;
<b>Similarity:</b> {course['Similarity']}%<br><br>
{course['Description']}
</div>
""",
                unsafe_allow_html=True
            )

# -------------------------------------------------
# Footer
# -------------------------------------------------
st.divider()

st.caption(
    "⚡ AI Recommendation System · TF-IDF + Cosine Similarity · Built with Streamlit"
)