# AI Image Summarization App

A local, private Streamlit app that describes images and answers questions about them using vision models running on [Ollama](https://ollama.com). Nothing leaves your machine.

## Features

- 📝 **Summary styles** — Detailed (two paragraphs, 500+ words), Brief (3 sentences), Bullet points, For kids, or Technical (composition, lighting, lens)
- ⚡ **Live streaming** — answers and summaries appear word by word as the model writes them, with a tokens-per-second readout
- 🔎 **Visual Q&A** — ask any question about what's in the picture
- ♿ **Alt text & caption** — accessibility alt text (under 125 characters), a social caption and hashtags, each with a copy button
- 🎨 **Colour palette** — the image's 3–10 dominant colours as hex swatches with their share, plus ready-to-copy CSS variables (computed locally, instant, no model needed)
- 🔊 **Read aloud** — play, pause and stop controls under every summary and answer, with 0.8–1.5× speed, using your browser's built-in voices (offline, nothing to install). Results stay on screen when you click other buttons, and summaries download as `.txt`
- 🟢 **Ollama health check** — the sidebar shows whether Ollama is running
- 🧠 **Auto model detection** — lists the vision models you actually have installed
- ⚡ **Auto-resize** — big phone photos are shrunk (and EXIF-rotated) before being sent to the model, for much faster responses; size is adjustable in the sidebar
- 🛟 **Friendly errors** — clear messages when Ollama is down, a model is missing, or a request times out

## Setup

```bash
pip install -r requirements.txt
ollama pull llava-phi3        # or moondream / qwen2.5vl
ollama serve                  # if it isn't already running
streamlit run app.py
```

Then open http://localhost:8501 and upload a JPG or PNG.

## Tips

- Smaller images and a lower **Max tokens** value give faster responses.
- `moondream` is the lightest model; `qwen2.5vl` is usually the most accurate.
