import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# Load model and vectorizer
model = joblib.load("modeltest.pkl")
vectorizer = joblib.load("vectorizertest.pkl")

# Label decoding
label_decode = {0: "Negative", 1: "Positive", 2: "Neutral"}
color_map = {
    "Positive": "#28a745",
    "Negative": "#dc3545",
    "Neutral": "#6c757d"
}

# Initialize history in session
if "history" not in st.session_state:
    st.session_state.history = []

# Sidebar
st.sidebar.title("🧠 Sentiment Analyzer")
st.sidebar.markdown("This app predicts the sentiment of your input text.")

# Show history
st.sidebar.markdown("---")
st.sidebar.subheader("📜 Prediction History")
if st.session_state.history:
    st.sidebar.dataframe(
        pd.DataFrame(st.session_state.history, columns=["Text", "Sentiment"]),
        use_container_width=True
    )
else:
    st.sidebar.write("No predictions made yet.")

# Main Panel
st.title("💬 Sentiment Analysis Web App")
st.markdown("Analyze the sentiment of your sentence using a trained machine learning model.")

# Text Input
user_input = st.text_area("✏️ Enter your sentence below:", height=100)

if st.button("🔍 Analyze Sentiment"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter some text to analyze.")
    else:
        # Predict
        vec = vectorizer.transform([user_input])
        prediction = model.predict(vec)[0]
        proba = model.predict_proba(vec)[0]

        sentiment = label_decode[prediction]
        color = color_map[sentiment]

        # Result
        st.markdown(
            f"### 🎯 Sentiment: **<span style='color:{color}'>{sentiment}</span>**",
            unsafe_allow_html=True
        )

        # Chart
        st.markdown("#### 📊 Model Confidence:")
        prob_df = pd.DataFrame({
            "Sentiment": ["Negative", "Positive", "Neutral"],
            "Probability": proba
        })

        fig, ax = plt.subplots()
        ax.bar(prob_df["Sentiment"], prob_df["Probability"], color=[color_map[s] for s in prob_df["Sentiment"]])
        ax.set_ylim(0, 1)
        ax.set_ylabel("Probability")
        ax.set_title("Model Confidence")
        st.pyplot(fig)

        # History
        st.session_state.history.insert(0, (user_input, sentiment))
