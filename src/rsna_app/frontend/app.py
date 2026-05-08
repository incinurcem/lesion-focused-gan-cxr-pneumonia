import io, base64, requests, streamlit as st
from PIL import Image

API_URL = "http://localhost:8000"
st.set_page_config(page_title="RSNA Multi-Model", page_icon="🫁", layout="wide")
st.title("🫁 RSNA Lesion — Multi-Model Comparison  (Leak-Free)")

@st.cache_data(ttl=60)
def fetch_models():
    return requests.get(f"{API_URL}/models", timeout=10).json()

@st.cache_data(ttl=60)
def fetch_samples(label, limit=100):
    return requests.get(f"{API_URL}/samples",
                        params={"label": label, "limit": limit},
                        timeout=15).json()

try:
    avail = fetch_models()
except Exception as e:
    st.error(f"Backend not responding: {e}")
    st.stop()

if not avail:
    st.error("No models found. Is MODELS_DIR correct?")
    st.stop()

st.sidebar.header("⚙️ Models")
opts = {f"{m['id']}  ({m['input_type']})": m["id"] for m in avail}
chosen = st.sidebar.multiselect("Models to compare",
                                 list(opts.keys()),
                                 default=list(opts.keys())[:2])
chosen_ids = [opts[c] for c in chosen]
mode = st.sidebar.radio("Image source", ["📋 From test set", "📤 Upload"])

def b64img(b):
    return Image.open(io.BytesIO(base64.b64decode(b)))

def render(results, true_label=None):
    if true_label is not None:
        tag = "POSITIVE" if true_label == 1 else "NEGATIVE"
        st.markdown(f"**Ground truth label:** `{true_label}`  ({tag})")
    for r in results:
        if "error" in r:
            st.error(f"`{r['model_id']}` → {r['error']}")
            continue
        st.markdown(f"### 🧠 `{r['model_id']}`  —  {r['arch']} / {r['input_type']}")
        c0, c1 = st.columns([1, 3])
        with c0:
            (st.error if r["probability"] > 0.5 else st.success)(r["prediction"])
            st.metric("Lesion probability", f"{r['probability']*100:.2f} %")
            st.progress(min(max(r["probability"], 0.0), 1.0))
            if "warning" in r:
                st.warning(r["warning"])
        with c1:
            cc = st.columns(3)
            cc[0].image(b64img(r["image_b64"]),   caption="Input",    use_container_width=True)
            cc[1].image(b64img(r["heatmap_b64"]), caption="Grad-CAM", use_container_width=True)
            cc[2].image(b64img(r["overlay_b64"]), caption="Overlay",  use_container_width=True)
        st.markdown("---")

if mode == "📋 From test set":
    lbl = st.sidebar.selectbox("Label filter", [1, 0],
                               format_func=lambda x: "Positive" if x == 1 else "Negative")
    samples = fetch_samples(lbl, 100)
    if not samples:
        st.warning("No samples found")
        st.stop()
    sid = st.selectbox(f"Select sample ({len(samples)})",
                        [s["sample_id"] for s in samples])
    if st.button("🚀 Compare", type="primary", disabled=not chosen_ids):
        with st.spinner("Running..."):
            r = requests.post(f"{API_URL}/predict_sample",
                              json={"sample_id": sid, "model_ids": chosen_ids},
                              timeout=180)
        if r.status_code == 200:
            d = r.json()
            render(d["results"], d["true_label"])
        else:
            st.error(f"{r.status_code}: {r.text}")
else:
    up = st.file_uploader("Upload chest X-ray", type=["png", "jpg", "jpeg"])
    if up:
        st.image(Image.open(up), width=300)
    if st.button("🚀 Compare", type="primary",
                 disabled=(up is None or not chosen_ids)):
        with st.spinner("Running..."):
            files = {"file": (up.name, up.getvalue(), up.type or "image/png")}
            r = requests.post(f"{API_URL}/predict_upload",
                              data={"model_ids": ",".join(chosen_ids)},
                              files=files, timeout=180)
        if r.status_code == 200:
            render(r.json()["results"])
        else:
            st.error(f"{r.status_code}: {r.text}")