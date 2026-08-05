import streamlit as st
import requests
import base64

OLLAMA_API = "http://localhost:11434/api/generate"

# Small, vision-capable models
MODEL_CHOICES = ["llava-phi3", "moondream", "qwen2.5vl"]  # prefer llava-phi3

def query_ollama(model: str, prompt: str, image_bytes: bytes, num_predict: int = 900, timeout=None):
    payload = {
        "model": model,
        "prompt": prompt,
        "images": [base64.b64encode(image_bytes).decode("utf-8")],
        "stream": False,
        "options": {
            "num_predict": num_predict,   # allow long output
            "temperature": 0.4,           # keep it focused/accurate
        },
    }
    resp = requests.post(OLLAMA_API, json=payload, timeout=timeout)
    resp.raise_for_status()
    return resp.json().get("response", "").strip()

st.set_page_config(page_title="Image Summary and question regarding the image", layout="centered")
st.title("Image Summary and question regarding the image")

# Sidebar controls
with st.sidebar:
    st.header("⚙️ Settings")
    model = st.selectbox("Model", MODEL_CHOICES, index=0)  # default llava-phi3
    max_tokens = st.slider("Max tokens (higher = longer, slower)", 200, 1400, 900, step=50)
    st.caption("Tip: For speed, use smaller images and reduce max tokens.")

uploaded = st.file_uploader("Upload an image (JPG/PNG)", type=["jpg", "jpeg", "png"])

if uploaded:
    image_bytes = uploaded.read()
    st.image(image_bytes, caption="Uploaded Image", use_container_width=True)

    # Q&A
    q = st.text_input("Ask a question about the image:")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔎 Get Answer"):
            with st.spinner("Answering..."):
                answer = query_ollama(
                    model,
                    f"Answer clearly and concisely based ONLY on the image. Question: {q}",
                    image_bytes,
                    num_predict=min(400, max_tokens),
                    timeout=None
                )
            st.subheader("Answer")
            st.write(answer)

    # Summary
    with col2:
        if st.button("📝 Summarize (2 paragraphs, ≥500 words)"):
            with st.spinner("Summarizing..."):
                prompt = (
                    "You are an expert vision describer. Look ONLY at the image. "
                    "Write a detailed description in EXACTLY TWO PARAGRAPHS totaling AT LEAST 500 words. "
                    "Be precise about objects, layout, colors, lighting, background, relationships, and possible context. "
                    "Avoid speculation beyond what is visible. Do not include lists or headings; use natural prose."
                )
            summary = query_ollama(
                model,
                prompt,
                image_bytes,
                num_predict=max_tokens,
                timeout=None
            )
            st.subheader("Summary")
            st.write(summary)
else:
    st.info("Upload an image to begin.")

