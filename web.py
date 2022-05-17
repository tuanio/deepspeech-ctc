import streamlit as st
import torch
import torchaudio
from streamlit.components.v1 import html
from hydra import compose, initialize
import hydra
from omegaconf import OmegaConf
from model import DeepSpeechModule
from utils import TextProcess
import matplotlib.pyplot as plt
import numpy as np

hydra.core.global_hydra.GlobalHydra.instance().clear()
initialize(config_path="conf", job_name="deepspeech")
cfg = compose(config_name="configs")
feature_transform = torchaudio.transforms.Spectrogram(n_fft=cfg.dataset.n_fft)


st.markdown(
    "<h1 style='text-align: center'>Speech Recognition of SOTA Team</h1>",
    unsafe_allow_html=True,
)
st.markdown(
    "<h4 style='text-align: center; text-decoration-line: line-through'><i>SOTA Team but not SOTA tools</i></h2>",
    unsafe_allow_html=True,
)

ckpt_option = st.selectbox(
    "Select the best of SOTA checkpoint ASR model",
    [
        "https://github.com/tuanio/deepspeech-ctc/releases/download/deepspeech/epoch.466-step.46204.ckpt",
        "https://github.com/tuanio/deepspeech-ctc/releases/download/deepspeech/epoch.555-step.54392.ckpt",
        "https://github.com/tuanio/deepspeech-ctc/releases/download/deepspeech/epoch.662-step.64236.ckpt",
        "https://github.com/tuanio/deepspeech-ctc/releases/download/deepspeech/epoch.767-step.73896.ckpt",
        "https://github.com/tuanio/deepspeech-ctc/releases/download/deepspeech/epoch.873-step.83648.ckpt",
        "https://github.com/tuanio/deepspeech-ctc/releases/download/deepspeech/epoch.978-step.102863.ckpt",
    ],
)

# for deepspeech
text_process = TextProcess(**cfg.text_process)
n_class = len(text_process.list_vocab)
model = DeepSpeechModule.load_from_checkpoint(
    ckpt_option,
    n_class=n_class,
    text_process=text_process,
    cfg_optim=cfg.optimizer,
    **cfg.model,
)

uploaded_file = st.file_uploader("Load file")
# audio = uploaded_file.read()
if uploaded_file is not None:
    # To read file as bytes:
    bytes_data = uploaded_file.getvalue()

    wave, sr = torchaudio.load(uploaded_file)
    fig, ax = plt.subplots(figsize=(12, 2))
    y = wave.numpy().reshape(-1)
    x = np.arange(y.shape[0]) / sr
    ax.plot(x, y, color="#323332")
    ax.set_ylabel("Amplitude")
    ax.set_xlabel("Time (s)")
    ax.grid(True, alpha=0.5, ls="-.")
    fig.tight_layout()
    st.pyplot(fig)

st.audio(uploaded_file, format="audio/wav")

if st.button("Predict"):

    specs = feature_transform(wave)  # channel, feature, time
    specs = specs.permute(0, 2, 1)  # channel, time, feature

    text = model(specs)[0]
    st.markdown(
        f"<h3 style='text-align: center'>{text}</h3>", unsafe_allow_html=True,
    )
