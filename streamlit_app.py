import streamlit as st
import torch
import numpy as np
import scipy.io.wavfile as wav
from transformers import AutoProcessor, MusicgenForConditionalGeneration
import tempfile

st.title("🎵 AI Prompt → Music Generator")

prompt = st.text_input("Enter music prompt")

if st.button("Generate Music"):

    st.write("Generating music... please wait")

    processor = AutoProcessor.from_pretrained("facebook/musicgen-small")
    model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-small")

    inputs = processor(
        text=[prompt],
        padding=True,
        return_tensors="pt"
    )

    audio_values = model.generate(**inputs, max_new_tokens=256)

    sampling_rate = model.config.audio_encoder.sampling_rate

    audio = audio_values[0,0].cpu().numpy()

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:

        wav.write(f.name, sampling_rate, audio)

        st.audio(f.name)

    st.success("Music generated successfully!")
