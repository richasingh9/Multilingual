# image_captioning_multilang.py
import streamlit as st
from PIL import Image
import io
import torch
from transformers import (
    VisionEncoderDecoderModel,
    ViTImageProcessor,
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    M2M100ForConditionalGeneration,
    M2M100Tokenizer,
)
st.set_page_config(page_title="Image Captioner - MultiLang", layout="centered")

st.title("Image Captioner — English + Multilingual (HI / BHO / TA)")
st.write("Upload an image → generate caption + description (English) → translate into selected languages.")

@st.cache_resource
def load_models():
    device = 0 if torch.cuda.is_available() else -1

    # 1) Image captioning model (short caption)
    caption_model_id = "nlpconnect/vit-gpt2-image-captioning"
    caption_model = VisionEncoderDecoderModel.from_pretrained(caption_model_id)
    feature_extractor = ViTImageProcessor.from_pretrained(caption_model_id)
    caption_tokenizer = AutoTokenizer.from_pretrained(caption_model_id)

    # 2) Text generator for longer description
    text_gen_model_id = "google/flan-t5-small"
    text_tokenizer = AutoTokenizer.from_pretrained(text_gen_model_id)
    text_gen_model = AutoModelForSeq2SeqLM.from_pretrained(text_gen_model_id)

    # 3) Multilingual translation: use M2M100 single model for many target languages
    trans_model_id = "facebook/m2m100_418M"
    trans_tokenizer = M2M100Tokenizer.from_pretrained(trans_model_id)
    trans_model = M2M100ForConditionalGeneration.from_pretrained(trans_model_id)

    # move heavy models to device (if available)
    if torch.cuda.is_available():
        caption_model.to("cuda")
        text_gen_model.to("cuda")
        trans_model.to("cuda")

    return {
        "caption_model": caption_model,
        "feature_extractor": feature_extractor,
        "caption_tokenizer": caption_tokenizer,
        "text_tokenizer": text_tokenizer,
        "text_gen_model": text_gen_model,
        "trans_tokenizer": trans_tokenizer,
        "trans_model": trans_model,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
    }

models = load_models()

# Language mapping: display name -> m2m100 language code
LANG_CODE = {
    "English": "en",
    "Hindi": "hi",
    "Bhojpuri": "bho",
    "Tamil": "ta",
}

st.sidebar.header("Translation targets")
selected_langs = st.sidebar.multiselect("Choose languages to generate (besides English):",
                                       options=list(LANG_CODE.keys()),
                                       default=["Hindi", "Bhojpuri"])

uploaded = st.file_uploader("Choose an image", type=["png", "jpg", "jpeg"])
if uploaded is not None:
    image = Image.open(io.BytesIO(uploaded.read())).convert("RGB")
    st.image(image, caption="Uploaded image", use_column_width=True)

    # --- Short caption (English)
    with st.spinner("Generating short caption (English)..."):
        pixel_values = models["feature_extractor"](images=image, return_tensors="pt").pixel_values
        pixel_values = pixel_values.to(models["device"])
        output_ids = models["caption_model"].generate(pixel_values, max_length=16, num_beams=4)
        caption_en = models["caption_tokenizer"].decode(output_ids[0], skip_special_tokens=True).strip()
    st.subheader("Caption (English)")
    st.write(caption_en)

    # --- Long description (English) using Flan-T5
    with st.spinner("Generating detailed description (English)..."):
        prompt = f"Write a detailed, descriptive paragraph about this image. Image caption: {caption_en}\nDescription:"
        inputs = models["text_tokenizer"].encode(prompt, return_tensors="pt", truncation=True, max_length=512)
        inputs = inputs.to(models["device"])
        outputs = models["text_gen_model"].generate(inputs, max_new_tokens=180, num_beams=4, early_stopping=True)
        description_en = models["text_tokenizer"].decode(outputs[0], skip_special_tokens=True).strip()
    st.subheader("Description (English)")
    st.write(description_en)

    # --- For each selected language, translate caption & description
    trans_toks = models["trans_tokenizer"]
    trans_model = models["trans_model"]
    translations = {}
    if selected_langs:
        with st.spinner("Translating to selected languages..."):
            for lang in selected_langs:
                code = LANG_CODE.get(lang)
                if code is None:
                    translations[lang] = {"caption": "Language code not found", "description": ""}
                    continue

                # m2m100: set tokenizer src lang to English and force target language id
                try:
                    trans_toks.src_lang = "en"
                    # tokenize caption
                    cap_enc = trans_toks(caption_en, return_tensors="pt", padding=True).to(models["device"])
                    forced_id = trans_toks.get_lang_id(code)
                    cap_gen = trans_model.generate(**cap_enc, forced_bos_token_id=forced_id, max_new_tokens=128)
                    caption_trans = trans_toks.decode(cap_gen[0], skip_special_tokens=True)

                    # tokenize description (may be long)
                    desc_enc = trans_toks(description_en, return_tensors="pt", padding=True, truncation=True, max_length=1024).to(models["device"])
                    desc_gen = trans_model.generate(**desc_enc, forced_bos_token_id=forced_id, max_new_tokens=512)
                    description_trans = trans_toks.decode(desc_gen[0], skip_special_tokens=True)

                    translations[lang] = {"caption": caption_trans, "description": description_trans}
                except Exception as e:
                    translations[lang] = {"caption": f"Translation error: {e}", "description": ""}

    # --- Display translations
    for lang in selected_langs:
        st.subheader(f"Caption ({lang})")
        st.write(translations.get(lang, {}).get("caption", "—"))
        st.subheader(f"Description ({lang})")
        st.write(translations.get(lang, {}).get("description", "—"))

    # --- Download results
    out_txt = []
    out_txt.append(f"Caption (EN): {caption_en}\n\nDescription (EN): {description_en}\n\n")
    for lang in selected_langs:
        out_txt.append(f"Caption ({lang}): {translations[lang]['caption']}\n\nDescription ({lang}): {translations[lang]['description']}\n\n")
    st.download_button("Download results (txt)", data="".join(out_txt), file_name="image_caption_multilang.txt")
else:
    st.write("Upload an image to generate captions and translations.")
