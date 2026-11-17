import base64
import io
import json
import re

import requests
import streamlit as st
from PIL import Image, ImageOps

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


# Summary styles: (button label, prompt, token budget cap)
BASE = "You are an expert vision describer. Look ONLY at the image and avoid speculation beyond what is visible. "
SUMMARY_STYLES = {
    "Detailed": ("2 paragraphs, ≥500 words",
                 BASE + "Write a detailed description in EXACTLY TWO PARAGRAPHS totaling AT LEAST 500 words. "
                 "Be precise about objects, layout, colors, lighting, background, relationships, and possible "
                 "context. Do not include lists or headings; use natural prose.", None),
    "Brief": ("3 sentences",
              BASE + "Describe the image in exactly 3 clear sentences: the main subject, the setting, and the "
                     "most notable detail.", 160),
    "Bullet points": ("key facts as bullets",
                      BASE + "List 6 to 10 short bullet points ('- ' at the start of each) covering the main "
                             "subject, people or objects, setting, colors, any visible text, and mood.", 350),
    "For kids": ("simple words",
                 BASE + "Describe the image for a 7-year-old in 4 to 5 short, friendly sentences using simple "
                        "words.", 220),
    "Technical": ("photography details",
                  BASE + "Analyse the image like a photographer: composition, framing, perspective, lighting "
                         "direction and quality, color palette, depth of field, and likely camera/lens choices. "
                         "Use short labelled paragraphs.", 600),
}


ALT_TEXT_PROMPT = (
    "Look ONLY at the image and write three things for publishing it online.\n"
    "ALT: one sentence of accessibility alt text, under 125 characters, describing what matters "
    "for someone who can't see it. Don't start with 'Image of' or 'Picture of'.\n"
    "CAPTION: one friendly social-media caption, under 20 words.\n"
    "TAGS: 5 relevant hashtags separated by spaces.\n"
    "Reply in exactly this format with the three labels ALT:, CAPTION: and TAGS:."
)


def parse_alt_text(raw: str) -> dict:
    """Pull ALT / CAPTION / TAGS out of the model reply; tolerant of small format slips."""
    out = {"alt": "", "caption": "", "tags": ""}
    current = None
    for line in raw.splitlines():
        line = line.replace("*", "").strip()  # models often bold the labels: **Alt text:**
        m = re.match(r"^(alt(?:\s*text)?|caption|tags|hashtags)\s*[:\-]\s*(.*)$", line, re.I)
        if m:
            label = m.group(1).lower()
            current = "alt" if label.startswith("alt") else ("tags" if "tag" in label else "caption")
            out[current] = m.group(2).strip()
        elif current and line.strip() and not out[current]:
            out[current] = line.strip()
    if not out["alt"]:
        out["alt"] = raw.strip().splitlines()[0] if raw.strip() else ""
    out["alt"] = re.sub(r"^(an?\s+)?(image|picture|photo)\s+of\s+", "", out["alt"], flags=re.I).strip().strip('"')
    if out["alt"]:
        out["alt"] = out["alt"][0].upper() + out["alt"][1:]
    tags = re.findall(r"#?([A-Za-z0-9_]+)", out["tags"])
    out["tags"] = " ".join("#" + t for t in tags[:8])
    return out


def prepare_image(raw: bytes, max_side: int):
    """Downscale so the longest side is at most max_side and re-encode as JPEG.

    Vision models resize internally anyway, so sending a 12 MP phone photo only
    makes the request slower (bigger base64 payload, more decoding). Also
    applies the EXIF rotation so portrait photos aren't sent sideways.
    Returns (bytes, (orig_w, orig_h), (new_w, new_h)).
    """
    img = Image.open(io.BytesIO(raw))
    img = ImageOps.exif_transpose(img)
    orig = img.size
    if max(orig) > max_side:
        img.thumbnail((max_side, max_side), Image.LANCZOS)
    if img.mode not in ("RGB", "L"):
        img = img.convert("RGB")
    out = io.BytesIO()
    img.save(out, format="JPEG", quality=90)
    return out.getvalue(), orig, img.size


def extract_palette(image_bytes: bytes, k: int = 6) -> list:
    """Dominant colours of the image, most common first: [(hex, share 0-1), ...].

    Runs locally with Pillow (median-cut quantisation on a thumbnail), so it is
    instant and needs no model. Near-identical colours are merged.
    """
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img.thumbnail((200, 200))
    q = img.quantize(colors=k * 2, method=Image.Quantize.MEDIANCUT)
    pal = q.getpalette()
    counts = sorted(q.getcolors(), reverse=True)  # [(count, index)]
    total = sum(c for c, _ in counts)
    picked = []
    for count, idx in counts:
        rgb = tuple(pal[idx * 3: idx * 3 + 3])
        for p in picked:
            if sum((a - b) ** 2 for a, b in zip(p["rgb"], rgb)) < 30 ** 2:
                p["n"] += count
                break
        else:
            picked.append({"rgb": rgb, "n": count})
    picked.sort(key=lambda p: -p["n"])
    return [("#%02x%02x%02x" % p["rgb"], p["n"] / total) for p in picked[:k]]


def text_colour(hex_code: str) -> str:
    r, g, b = (int(hex_code[i:i + 2], 16) for i in (1, 3, 5))
    return "#000" if 0.299 * r + 0.587 * g + 0.114 * b > 150 else "#fff"


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


def stream_ollama(model: str, prompt: str, image_bytes: bytes, num_predict: int = 900, stats=None, timeout=300):
    """Yield the model's answer piece by piece as Ollama generates it.

    If `stats` is a dict, it is filled with token count and speed when done.
    """
    payload = {
        "model": model,
        "prompt": prompt,
        "images": [base64.b64encode(image_bytes).decode("utf-8")],
        "stream": True,
        "options": {"num_predict": num_predict, "temperature": 0.4},
    }
    with requests.post(OLLAMA_API, json=payload, stream=True, timeout=timeout) as resp:
        resp.raise_for_status()
        for line in resp.iter_lines():
            if not line:
                continue
            chunk = json.loads(line)
            if chunk.get("error"):
                raise RuntimeError(chunk["error"])
            if chunk.get("response"):
                yield chunk["response"]
            if chunk.get("done"):
                if stats is not None:
                    tokens = chunk.get("eval_count") or 0
                    secs = (chunk.get("eval_duration") or 0) / 1e9
                    stats.update(tokens=tokens, seconds=secs,
                                 tps=tokens / secs if secs else 0.0)
                break


def run_stream(model, prompt, image_bytes, num_predict):
    """Show the answer live as it is written; returns the full text or None."""
    stats = {}
    try:
        text = st.write_stream(stream_ollama(model, prompt, image_bytes, num_predict, stats=stats))
    except requests.ConnectionError:
        st.error("Can't reach Ollama. Start it with `ollama serve` and try again.")
        return None
    except requests.Timeout:
        st.error("The model took too long to respond. Try a smaller image or fewer max tokens.")
        return None
    except requests.HTTPError as e:
        detail = e.response.text if e.response is not None else str(e)
        st.error(f"Ollama returned an error: {detail}\n\nIs the model pulled? Try `ollama pull {model}`.")
        return None
    except RuntimeError as e:
        st.error(f"Ollama returned an error: {e}")
        return None
    text = (text or "").strip() if isinstance(text, str) else "".join(map(str, text or [])).strip()
    if stats.get("tokens"):
        st.caption(f"⚡ {stats['tokens']} tokens · {stats['tps']:.1f} tokens/s")
    return text or None


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
    max_side = st.select_slider(
        "Max image size sent to the model (px)",
        options=[512, 768, 1024, 1536, 2048],
        value=1024,
        help="Large photos are shrunk before sending — much faster with no real loss of detail for the model.",
    )
    st.caption("Tip: For speed, lower the image size and max tokens.")

uploaded = st.file_uploader("Upload an image (JPG/PNG)", type=["jpg", "jpeg", "png"])

if uploaded:
    raw = uploaded.getvalue()  # read() returns b"" on reruns
    try:
        image_bytes, orig_size, new_size = prepare_image(raw, max_side)
    except Exception:
        st.error("Couldn't read that image. Please upload a valid JPG or PNG.")
        st.stop()
    st.image(image_bytes, caption="Uploaded Image", use_container_width=True)
    if new_size != orig_size:
        st.caption(
            f"Resized {orig_size[0]}×{orig_size[1]} → {new_size[0]}×{new_size[1]} "
            f"({len(raw) / 1024:.0f} KB → {len(image_bytes) / 1024:.0f} KB) for faster answers."
        )

    # Q&A
    q = st.text_input("Ask a question about the image:")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔎 Get Answer", disabled=not q.strip()):
            st.subheader("Answer")
            run_stream(
                model,
                f"Answer clearly and concisely based ONLY on the image. Question: {q}",
                image_bytes,
                min(400, max_tokens),
            )

    # Alt text
    if st.button("♿ Alt text & caption"):
        with st.spinner("Writing alt text..."):
            raw_alt = run_query(model, ALT_TEXT_PROMPT, image_bytes, num_predict=200)
        if raw_alt:
            alt = parse_alt_text(raw_alt)
            st.subheader("Alt text")
            st.code(alt["alt"], language=None)   # code block = one-click copy
            n = len(alt["alt"])
            st.caption(f"{n} characters" + (" — screen readers work best under 125" if n > 125 else " ✓"))
            if alt["caption"]:
                st.subheader("Caption")
                st.code(alt["caption"], language=None)
            if alt["tags"]:
                st.subheader("Hashtags")
                st.code(alt["tags"], language=None)

    # Colour palette (local, no model needed)
    with st.expander("🎨 Colour palette"):
        n_colours = st.slider("Colours", 3, 10, 6)
        palette = extract_palette(image_bytes, n_colours)
        st.markdown(
            '<div style="display:flex;border-radius:8px;overflow:hidden;height:72px">' + "".join(
                f'<div style="flex:{share:.3f};background:{hx};color:{text_colour(hx)};display:flex;'
                f'align-items:flex-end;padding:4px;font:12px monospace">{hx}<br>{share:.0%}</div>'
                for hx, share in palette) + "</div>",
            unsafe_allow_html=True,
        )
        st.code(":root {\n" + "".join(f"  --color-{i + 1}: {hx};\n" for i, (hx, _) in enumerate(palette)) + "}",
                language="css")
        st.caption("Copy the CSS variables, or use the hex codes in Figma/Canva for a matching design.")

    # Summary
    with col2:
        style = st.selectbox("Summary style", list(SUMMARY_STYLES), index=0,
                             format_func=lambda s: f"{s} — {SUMMARY_STYLES[s][0]}")
        if st.button(f"📝 Summarize ({style.lower()})"):
            _, prompt, cap = SUMMARY_STYLES[style]
            st.subheader("Summary")
            summary = run_stream(model, prompt, image_bytes, min(cap or max_tokens, max_tokens))
            if summary:
                st.caption(f"{len(summary.split())} words")
else:
    st.info("Upload an image to begin.")
