import streamlit as st
import requests
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
HF_API_KEY = os.getenv("HF_API_TOKEN")

# API URL and headers
API_URL = "https://api-inference.huggingface.co/models/openai/whisper-large-v3"
headers = {"Authorization": f"Bearer {HF_API_KEY}"}

# Function to send query to the model
def query_model(file, task="transcribe"):
    response = requests.post(API_URL, headers=headers, data=file, params={"task": task})
    return response.json()

# Function to get supported languages for translation
def get_supported_languages():
    return {
        "English": "en", "Spanish": "es", "French": "fr", "German": "de", "Italian": "it",
        "Chinese": "zh", "Japanese": "ja", "Korean": "ko", "Russian": "ru", "Portuguese": "pt"
    }

# Streamlit UI with tabs
st.title("🌍 Whisper Large V3: Speech-to-Text & Translation")
st.write("Upload an audio file and explore different functionalities like transcription, language detection, and translation!")

# File uploader for audio files
uploaded_file = st.file_uploader("Upload an audio file (.flac, .wav, .mp3)", type=["flac", "wav", "mp3"])

# Create tabs for multiple use cases
tab1, tab2, tab3 = st.tabs(["🎙️ Transcribe", "🌍 Translate", "🔍 Language Detection"])

with tab1:
    st.header("🎙️ Transcribe Audio to Text")
    
    if uploaded_file is not None:
        file_bytes = uploaded_file.read()

        if st.button("Transcribe Audio"):
            with st.spinner("Transcribing..."):
                transcription = query_model(file_bytes)
                if "text" in transcription:
                    st.success("Transcription Complete!")
                    st.text_area("Transcribed Text", transcription["text"], height=300)
                    
                    # Download button for the transcription text
                    st.download_button(
                        label="💾 Download Transcription",
                        data=transcription["text"],
                        file_name=f"transcription_{uploaded_file.name}.txt",
                        mime="text/plain"
                    )
                else:
                    st.error("Transcription failed. Please try again.")
    else:
        st.warning("Please upload an audio file to transcribe.")

with tab2:
    st.header("🌍 Translate Audio into Another Language")
    
    # Select target language for translation
    languages = get_supported_languages()
    target_language = st.selectbox("Select the target language", list(languages.keys()))

    if uploaded_file is not None:
        file_bytes = uploaded_file.read()

        if st.button(f"Translate to {target_language}"):
            with st.spinner(f"Translating to {target_language}..."):
                translation = query_model(file_bytes, task="translate")
                if "text" in translation:
                    st.success(f"Translation to {target_language} Complete!")
                    st.text_area(f"Translated Text in {target_language}", translation["text"], height=300)
                    
                    # Download button for the translated text
                    st.download_button(
                        label="💾 Download Translation",
                        data=translation["text"],
                        file_name=f"translation_{uploaded_file.name}_{target_language}.txt",
                        mime="text/plain"
                    )
                else:
                    st.error("Translation failed. Please try again.")
    else:
        st.warning("Please upload an audio file to translate.")

with tab3:
    st.header("🔍 Detect Language in Audio")

    if uploaded_file is not None:
        file_bytes = uploaded_file.read()

        if st.button("Detect Language"):
            with st.spinner("Detecting language..."):
                detection = query_model(file_bytes, task="language-detection")
                if "language" in detection:
                    st.success(f"Detected Language: {detection['language'].title()}")
                else:
                    st.error("Language detection failed. Please try again.")
    else:
        st.warning("Please upload an audio file for language detection.")

# Footer for tips and suggestions
st.markdown("""
    ---
    **Usage Tips:**
    - Use clear audio for better transcription accuracy.
    - Choose different languages for a fun translation experience!
    - Experiment with different types of audio for language detection.
""")
