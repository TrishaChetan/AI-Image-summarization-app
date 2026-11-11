# AI Image Summarization App

A local, private Streamlit app that describes images and answers questions about them using vision models running on [Ollama](https://ollama.com). Nothing leaves your machine.

## Features

- 📝 **Detailed summary** — two-paragraph, 500+ word description of the image
- 🔎 **Visual Q&A** — ask any question about what's in the picture
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
