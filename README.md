# Sentiment Analysis Web App

This project is a Streamlit-based web application for sentiment analysis using both text and voice input. It leverages a trained machine learning model to classify input as Positive, Negative, or Neutral.

## Features
- **Text Input:** Enter any sentence to analyze its sentiment.
- **Voice Input:** Use your microphone to speak and transcribe text for sentiment analysis.
- **Model Confidence:** Visualizes the model's confidence for each sentiment class.
- **Prediction History:** View a sidebar history of all predictions made in the session.

## Files
- `speech.py`: Main application script.
- `modeltest.pkl`: Trained sentiment analysis model (pickle file).
- `vectorizertest.pkl`: Text vectorizer (pickle file).
- `sentiment_dataset_1000.csv`: Dataset used for training/testing (optional).
- `requirements.txt`: List of required Python packages.

## Setup & Usage
1. **Install dependencies:**
   ```powershell
   pip install -r requirements.txt
   ```
2. **Run the app:**
   ```powershell
   streamlit run speech.py
   ```
3. **Interact:**
   - Use the sidebar to input text or voice.
   - View sentiment prediction and model confidence.

## Requirements
- Python 3.7+
- See `requirements.txt` for all dependencies (includes `streamlit`, `scikit-learn`, `joblib`, `speechrecognition`, `matplotlib`, `pandas`).

## Notes
- Ensure your microphone is connected for voice input.
- The model and vectorizer files (`modeltest.pkl`, `vectorizertest.pkl`) must be present in the project directory.

## License
This project is for educational purposes.
