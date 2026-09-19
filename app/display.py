import streamlit as st
import joblib
import os

# ============================================================
# 1. Page Configuration
# ============================================================

st.set_page_config(
    page_title="Review Rating Predictor",
    page_icon="⭐",
    layout="centered"
)

# ============================================================
# 2. Load Saved Model and TF-IDF Vectorizer
# ============================================================

# Get the project root directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Model files inside the models folder
model_path = os.path.join(
    BASE_DIR,
    "models",
    "best_overall__model.pkl"
)

vectorizer_path = os.path.join(
    BASE_DIR,
    "models",
    "tfidf_vectorizer__best.pkl"
)


@st.cache_resource
def load_artifacts():
    if os.path.exists(model_path) and os.path.exists(vectorizer_path):
        model = joblib.load(model_path)
        vectorizer = joblib.load(vectorizer_path)
        return model, vectorizer

    return None, None


model, vectorizer = load_artifacts()

# ============================================================
# 3. Streamlit User Interface
# ============================================================

st.title("⭐ Product Review Rating Predictor")

st.write(
    "Enter your product review below, and our machine learning "
    "model will predict the rating score!"
)

user_review = st.text_area(
    "Your Review:",
    placeholder=(
        "Type something like: "
        "'This product exceeded my expectations! Absolutely love it.'"
    )
)

# ============================================================
# 4. Prediction
# ============================================================

if st.button("Predict Rating", type="primary"):

    if not user_review.strip():
        st.warning("Please enter a valid review text before predicting.")

    elif model is None or vectorizer is None:
        st.error(
            "Model or vectorizer files not found in the 'models' folder! "
            "Please run your training script first."
        )

    else:
        # Convert review text into TF-IDF features
        review_vec = vectorizer.transform([user_review])

        # Predict rating
        prediction = model.predict(review_vec)

        # Display predicted rating
        st.success(f"### Predicted Star Rating: {prediction[0]} ⭐")

        # Dynamic feedback
        score = int(prediction[0])

        if score >= 4:
            st.balloons()
            st.info("This looks like a highly positive review.")

        elif score == 3:
            st.info("This looks like a neutral review.")

        else:
            st.warning("This looks like a negative review.")

# ============================================================
# 5. Footer
# ============================================================

st.markdown("---")
st.caption("Powered by Scikit-Learn, TF-IDF, and Streamlit.")



# streamlit run frontend/display.py
