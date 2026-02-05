# Streamlit app: Image -> Caption + Description (English + Hindi)

import streamlit as st
from PIL import Image
import io
import os
import torch

from transformers import (
    VisionEncoderDecoderModel,
    ViTImageProcessor,
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    pipeline,
)

# -------------------- CONFIG --------------------
st.set_page_config(
    page_title="Image Captioner (EN + HI)",
    layout="centered"
)

st.title("🖼 Image Captioner — English + Hindi")
st.write(
    "Upload an image. The app generates a short caption and a detailed description "
    "in English and translates both into Hindi."
)

# Secure HF token (Streamlit Secrets)
os.environ["HF_TOKEN"] = st.secrets["HF_TOKEN"]

DEVICE = "cpu"   # Streamlit Cloud SAFE

# -------------------- LOAD MODELS --------------------
@st.cache_resource
def load_models():
    # 1️⃣ Image Captioning model
    caption_model_id = "nlpconnect/vit-gpt2-image-captioning"
    caption_model = VisionEncoderDecoderModel.from_pretrained(caption_model_id)
    caption_model.to(DEVICE)

    feature_extractor = ViTImageProcessor.from_pretrained(caption_model_id)
    caption_tokenizer = AutoTokenizer.from_pretrained(caption_model_id)

    # 2️⃣ Text generation (detailed description)
    text_gen_model_id = "google/flan-t5-small"
    text_gen_model = AutoModelForSeq2SeqLM.from_pretrained(text_gen_model_id)
    text_gen_model.to(DEVICE)

    text_tokenizer = AutoTokenizer.from_pretrained(text_gen_model_id)

    # 3️⃣ Translation EN -> HI (CPU forced)
    translation = pipeline(
        "translation_en_to_hi",
        model="Helsinki-NLP/opus-mt-en-hi",
        device=-1   # CPU only
    )

    return {
        "caption_model": caption_model,
        "feature_extractor": feature_extractor,
        "caption_tokenizer": caption_tokenizer,
        "text_gen_model": text_gen_model,
        "text_tokenizer": text_tokenizer,
        "translation": translation,
    }

models = load_models()

# -------------------- UI --------------------
uploaded = st.file_uploader(
    "📤 Upload an image",
    type=["png", "jpg", "jpeg"]
)

if uploaded:
    image = Image.open(io.BytesIO(uploaded.read())).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # ---------- SHORT CAPTION ----------
    pixel_values = models["feature_extractor"](
        images=image,
        return_tensors="pt"
    ).pixel_values.to(DEVICE)

    with st.spinner("🧠 Generating short caption..."):
        output_ids = models["caption_model"].generate(
            pixel_values,
            max_length=16,
            num_beams=4
        )
        caption_en = models["caption_tokenizer"].decode(
            output_ids[0],
            skip_special_tokens=True
        ).strip()

    st.subheader("📝 Caption (English)")
    st.write(caption_en)

    # ---------- DETAILED DESCRIPTION ----------
    prompt = (
        "Write a detailed, vivid, and informative paragraph describing the image.\n\n"
        f"Image caption: {caption_en}\n\nDescription:"
    )

    inputs = models["text_tokenizer"].encode(
        prompt,
        return_tensors="pt"
    ).to(DEVICE)

    with st.spinner("📖 Generating detailed description (English)..."):
        outputs = models["text_gen_model"].generate(
            inputs,
            max_new_tokens=200,
            num_beams=4,
            early_stopping=True
        )
        description_en = models["text_tokenizer"].decode(
            outputs[0],
            skip_special_tokens=True
        ).strip()

    st.subheader("📘 Description (English)")
    st.write(description_en)

    # ---------- TRANSLATION ----------
    with st.spinner("🌐 Translating to Hindi..."):
        caption_hi = models["translation"](caption_en)[0]["translation_text"]
        description_hi = models["translation"](description_en)[0]["translation_text"]

    st.subheader("📝 Caption (Hindi)")
    st.write(caption_hi)

    st.subheader("📘 Description (Hindi)")
    st.write(description_hi)

    # ---------- DOWNLOAD ----------
    result_text = f"""
Caption (English):
{caption_en}

Description (English):
{description_en}

Caption (Hindi):
{caption_hi}

Description (Hindi):
{description_hi}
"""

    st.download_button(
        "⬇ Download Results (TXT)",
        data=result_text,
        file_name="image_caption_results.txt"
    )

else:
    st.info("👆 Upload an image to start captioning.")
