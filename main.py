# main.py - IMDB SENTIMENT ANALYSIS (Portfolio Version)
import os
import re
import streamlit as st
from joblib import load

# ------------------- PROJECT OVERVIEW SIDEBAR -------------------
with st.sidebar:
    st.markdown("## 🎯 **Project Overview**")
    st.markdown("""
    **IMDB Movie Review Sentiment Analysis**

  
    **🔄 Full Pipeline**:
    1. Kaggle IMDB dataset download
    2. Text preprocessing (NLTK)
    3. TF-IDF vectorization (Scikit-learn)
    4. Train/test split (80/20, stratified)
    5. Model training & evaluation
    6. Joblib persistence (production-ready)

    **🛠️ Tech Stack**:
    - Python | Scikit-learn | NLTK
    - Streamlit | Joblib | Pandas

    ---
    st.markdown(
    "[GitHub repository](https://github.com/anushri0309/Movie-Sentiment-Analysis)"
)

    [**LinkedIn Post**]()
    """)


# ------------------- LOAD MODELS -------------------
@st.cache_resource
def load_models():
    model_path = os.path.join("models", "model.joblib")
    vect_path = os.path.join("models", "vectorizer.joblib")

    if not os.path.exists(model_path) or not os.path.exists(vect_path):
        st.error("❌ **Model files not found!** Run `python train_model.py` first.")
        st.stop()

    model = load(model_path)
    vectorizer = load(vect_path)
    return model, vectorizer


model, vectorizer = load_models()


# ------------------- CLEAN TEXT FUNCTION -------------------
def clean_review(text):
    """Match training preprocessing exactly"""
    text = re.sub("[^a-zA-Z]", " ", text)
    text = text.lower()
    text = text.split()
    return " ".join(text)


# ------------------- MAIN UI -------------------
st.set_page_config(
    page_title="IMDB Sentiment Analysis",
    page_icon="🎬",
    layout="wide"
)

st.title("🎬 **IMDB Movie Review Sentiment Analysis**")
st.markdown("*Production-ready ML pipeline with 89% accuracy*")

# Two-column layout
col1, col2 = st.columns([3, 1])

with col1:
    st.markdown("### **📝 Enter Movie Review**")
    review = st.text_area(
        "",
        height=250,
        placeholder="Paste any movie review here...\n\nExample: 'This movie was absolutely wonderful with great acting and a gripping storyline!'"
    )

    if st.button("🎭 **Analyze Sentiment**", type="primary", use_container_width=True):
        if review.strip():
            # Predict
            cleaned = clean_review(review)
            prediction = model.predict(vectorizer.transform([cleaned]))[0]
            confidence = model.predict_proba(vectorizer.transform([cleaned]))[0].max()

            # Results
            st.markdown("### **✅ Prediction**")
            if prediction == 1:
                st.markdown("## 🟢 **POSITIVE**")
                st.balloons()
            else:
                st.markdown("## 🔴 **NEGATIVE**")

            st.markdown(f"**Confidence**: {confidence:.1%}")
            st.markdown(f"**Review length**: {len(review.split())} words")

        else:
            st.warning("📝 **Please enter a movie review first!**")

with col2:
    st.markdown("### **📈 Model Performance**")
    st.success("**Test Accuracy: 89.0%**")
    st.info("✅ Trained on 5K real IMDB reviews")
    st.info("✅ 80/20 train/test split")
    st.info("✅ TF-IDF (5K features)")

    st.markdown("---")
    st.markdown("### **🎯 What it predicts**")
    st.markdown("- 🟢 **Positive**: great, amazing, loved, recommend")
    st.markdown("- 🔴 **Negative**: boring, waste, terrible, hate")

# Footer
st.markdown("---")

