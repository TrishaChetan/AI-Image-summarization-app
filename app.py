import base64

import requests
import streamlit as st

OLLAMA_HOST = "http://localhost:11434"
OLLAMA_API = f"{OLLAMA_HOST}/api/generate"

# Small, vision-capable models (used when Ollama can't be queried for its list)
DEFAULT_MODELS = ["llava-phi3", "moondream", "qwen2.5vl"]
VISION_HINTS = ("llava", "moondream", "qwen2.5vl", "qwen-vl", "bakllava", "minicpm-v", "llama3.2-vision", "gemma3")


@st.cache_data(ttl=30)
def ollama_status():
    """Return (is_running, installed_vision_models)."""
    try:
        resp = requests.get(f"{OLLAMA_HOST}/api/tags", timeout=3)
        resp.raise_for_status()
    except requests.RequestException:
        return False, []
    names = [m.get("name", "") for m in resp.json().get("models", [])]
    vision = [n for n in names if any(h in n.lower() for h in VISION_HINTS)]
    return True, vision


def query_ollama(model: str, prompt: str, image_bytes: bytes, num_predict: int = 900, timeout=300):
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


def run_query(model, prompt, image_bytes, num_predict):
    """Call the model and show a friendly error instead of a stack trace."""
    try:
        return query_ollama(model, prompt, image_bytes, num_predict=num_predict)
    except requests.ConnectionError:
        st.error("Can't reach Ollama. Start it with `ollama serve` and try again.")
    except requests.Timeout:
        st.error("The model took too long to respond. Try a smaller image or fewer max tokens.")
    except requests.HTTPError as e:
        detail = e.response.text if e.response is not None else str(e)
        st.error(f"Ollama returned an error: {detail}\n\nIs the model pulled? Try `ollama pull {model}`.")
    return None


st.set_page_config(page_title="Image Summary and question regarding the image", layout="centered")
st.title("Image Summary and question regarding the image")

running, installed = ollama_status()

# Sidebar controls
with st.sidebar:
    st.header("⚙️ Settings")
    if running:
        st.success("Ollama is running")
    else:
        st.error("Ollama is not reachable at localhost:11434")
    choices = installed or DEFAULT_MODELS
    model = st.selectbox("Model", choices, index=0)
    if running and not installed:
        st.caption("No vision models detected — pull one with `ollama pull llava-phi3`.")
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
        if st.button("🔎 Get Answer", disabled=not q.strip()):
            with st.spinner("Answering..."):
                answer = run_query(
                    model,
                    f"Answer clearly and concisely based ONLY on the image. Question: {q}",
                    image_bytes,
                    num_predict=min(400, max_tokens),
                )
            if answer:
                st.subheader("Answer")
                st.write(answer)

    # Summary
    with col2:
        if st.button("📝 Summarize (2 paragraphs, ≥500 words)"):
            prompt = (
                "You are an expert vision describer. Look ONLY at the image. "
                "Write a detailed description in EXACTLY TWO PARAGRAPHS totaling AT LEAST 500 words. "
                "Be precise about objects, layout, colors, lighting, background, relationships, and possible context. "
                "Avoid speculation beyond what is visible. Do not include lists or headings; use natural prose."
            )
            with st.spinner("Summarizing..."):
                summary = run_query(model, prompt, image_bytes, num_predict=max_tokens)
            if summary:
                st.subheader("Summary")
                st.write(summary)
                st.caption(f"{len(summary.split())} words")
else:
    st.info("Upload an image to begin.")
