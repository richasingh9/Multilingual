# Streamlit app: Image -> Caption + Description (English + Hindi)
# Requirements:
# pip install streamlit transformers torch pillow sentencepiece

import streamlit as st
from PIL import Image
import io
import torch
from transformers import (
    VisionEncoderDecoderModel,
    ViTImageProcessor,
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    pipeline,
)

st.set_page_config(page_title="Image Captioner (EN + HI)", layout="centered")
st.title("Image captioner — English + Hindi")
st.write("Upload an image. The app generates a short caption and a longer description in English and translates both to Hindi.")

@st.cache_resource
def load_models():
    device = 0 if torch.cuda.is_available() else -1

    # 1) Image captioning (short caption)
    caption_model_id = "nlpconnect/vit-gpt2-image-captioning"
    caption_model = VisionEncoderDecoderModel.from_pretrained(caption_model_id)
    feature_extractor = ViTImageProcessor.from_pretrained(caption_model_id)
    caption_tokenizer = AutoTokenizer.from_pretrained(caption_model_id)

    # 2) Text generator for longer description (Flan-T5 small — lightweight)
    text_gen_model_id = "google/flan-t5-small"
    text_tokenizer = AutoTokenizer.from_pretrained(text_gen_model_id)
    text_gen_model = AutoModelForSeq2SeqLM.from_pretrained(text_gen_model_id)

    # 3) Translation pipeline EN -> HI
    translation = pipeline("translation_en_to_hi", model="Helsinki-NLP/opus-mt-en-hi")

    return {
        "caption_model": caption_model,
        "feature_extractor": feature_extractor,
        "caption_tokenizer": caption_tokenizer,
        "text_tokenizer": text_tokenizer,
        "text_gen_model": text_gen_model,
        "translation": translation,
        "device": device,
    }

models = load_models()

uploaded = st.file_uploader("Choose an image", type=["png", "jpg", "jpeg"])
if uploaded is not None:
    image = Image.open(io.BytesIO(uploaded.read())).convert("RGB")
    st.image(image, caption="Uploaded image", use_column_width=True)

    # 1) Short caption (image -> short caption)
    pixel_values = models["feature_extractor"](images=image, return_tensors="pt").pixel_values
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models["caption_model"].to(device)
    pixel_values = pixel_values.to(device)

    with st.spinner("Generating caption..."):
        output_ids = models["caption_model"].generate(pixel_values, max_length=16, num_beams=4)
        caption = models["caption_tokenizer"].decode(output_ids[0], skip_special_tokens=True).strip()

    st.subheader("Caption (English)")
    st.write(caption)

    # 2) Longer description: use the caption as prompt to text generator
    prompt = f"Write a detailed, descriptive paragraph about this image. Keep it informative and vivid.\nImage caption: {caption}\nDescription:"
    inputs = models["text_tokenizer"].encode(prompt, return_tensors="pt")
    inputs = inputs.to(device)
    models["text_gen_model"].to(device)

    with st.spinner("Generating detailed description (English)..."):
        outputs = models["text_gen_model"].generate(inputs, max_new_tokens=300, num_beams=4, early_stopping=True)
        description_en = models["text_tokenizer"].decode(outputs[0], skip_special_tokens=True).strip()

    st.subheader("Description (English)")
    st.write(description_en)

    # 3) Translate both to Hindi
    with st.spinner("Translating to Hindi..."):
        caption_hi = models["translation"](caption)[0]["translation_text"]
        description_hi = models["translation"](description_en)[0]["translation_text"]

    st.subheader("Caption (Hindi)")
    st.write(caption_hi)

    st.subheader("Description (Hindi)")
    st.write(description_hi)

    st.info("Notes: Models run faster on a GPU. flan-t5-small and vit-gpt2 are moderate-size; change to larger models for better quality.\nIf memory errors occur, try using smaller models or run on a machine with more RAM/GPU.")

    st.download_button("Download results (txt)", data=(f"Caption (EN): {caption}\n\nDescription (EN): {description_en}\n\nCaption (HI): {caption_hi}\n\nDescription (HI): {description_hi}"), file_name="image_caption_results.txt")

else:
    st.write("Waiting for an image — upload a photo to try it out.")
