# image_captioning_multilang_improved.py
import streamlit as st
from PIL import Image
import io
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from transformers import (
    VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer, AutoModelForSeq2SeqLM,
    M2M100ForConditionalGeneration, M2M100Tokenizer,
    AutoModelForCausalLM, AutoTokenizer as AutoTok
)

st.set_page_config(page_title="Image Captioner - Improved Multilang", layout="centered")
st.title("Image Captioner — EN → HI / TA / BHO (improved)")

@st.cache_resource
def load_models():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # caption model (same)
    #cap_id = "Salesforce/blip-image-captioning-base"
    #cap_model = VisionEncoderDecoderModel.from_pretrained(cap_id)
    #feat = ViTImageProcessor.from_pretrained(cap_id)
    #cap_tok = AutoTokenizer.from_pretrained(cap_id)

    
    cap_id = "Salesforce/blip2-flan-t5-xl"
    processor = BlipProcessor.from_pretrained(cap_id)
    cap_model = BlipForConditionalGeneration.from_pretrained(cap_id)

    # text generator
    text_id = "google/flan-t5-base"
    text_tok = AutoTokenizer.from_pretrained(text_id)
    text_model = AutoModelForSeq2SeqLM.from_pretrained(text_id)

    # Hindi translator (Helsinki small & reliable)
    hi_model_id = "Helsinki-NLP/opus-mt-en-hi"
    hi_tok = AutoTokenizer.from_pretrained(hi_model_id)
    hi_model = AutoModelForSeq2SeqLM.from_pretrained(hi_model_id)

    # Tamil translator (M2M100)
    m2m_id = "facebook/m2m100_418M"
    m2m_tok = M2M100Tokenizer.from_pretrained(m2m_id)
    m2m_model = M2M100ForConditionalGeneration.from_pretrained(m2m_id)

    # Bhojpuri LM (fallback paraphraser) - optional, may not exist in all setups
    bho_tokenizer = None
    bho_model = None
    try:
        bho_tok = AutoTok.from_pretrained("pksx01/sarvam-1-it-bhojpuri")
        bho_m = AutoModelForCausalLM.from_pretrained("pksx01/sarvam-1-it-bhojpuri")
        bho_tokenizer = bho_tok
        bho_model = bho_m
    except Exception:
        # If unavailable, we'll fallback to showing Hindi result
        bho_tokenizer = None
        bho_model = None

    # Move big models to device
    if device == "cuda":
        cap_model.to(device)
        text_model.to(device)
        hi_model.to(device)
        m2m_model.to(device)
        if bho_model:
            bho_model.to(device)

    return {
        "device": device,
        "cap_model": cap_model, "processor": processor,
        "text_model": text_model, "text_tok": text_tok,
        "hi_model": hi_model, "hi_tok": hi_tok,
        "m2m_model": m2m_model, "m2m_tok": m2m_tok,
        "bho_model": bho_model, "bho_tok": bho_tokenizer
    }

models = load_models()

LANGS = ["Hindi", "Tamil", "Bhojpuri", "English"]
sel = st.multiselect("Choose languages to output", options=LANGS, default=["Hindi","Tamil","Bhojpuri","English"])

uploaded = st.file_uploader("Upload image", type=["jpg","jpeg","png"])
if uploaded:
    image = Image.open(io.BytesIO(uploaded.read())).convert("RGB")
    st.image(image, use_column_width=True)

    # caption en
    #px = models["feat"](images=image, return_tensors="pt").pixel_values.to(models["device"])
    #out = models["cap_model"].generate(px, max_length=16, num_beams=4)
    #cap_en = models["cap_tok"].decode(out[0], skip_special_tokens=True).strip()
    inputs = models["processor"](image, return_tensors="pt").to(models["device"])
    out = models["cap_model"].generate(**inputs)
    cap_en = models["processor"].decode(out[0], skip_special_tokens=True)
    st.subheader("Caption (English)"); st.write(cap_en)

    # description en
    prompt = f"Write a single paragraph of about 50 words describing the image in detail, including objects, actions, colors, and surroundings: {cap_en}"
    inputs = models["text_tok"].encode(prompt, return_tensors="pt", truncation=True, max_length=512).to(models["device"])
    desc_ids = models["text_model"].generate(inputs, max_new_tokens=80, do_sample=True, temperature=0.7, top_p=0.9)
    desc_en = models["text_tok"].decode(desc_ids[0], skip_special_tokens=True).strip()
    st.subheader("Description (English)"); st.write(desc_en)
    desc_en = desc_en.replace("\n", " ")
    desc_en = " ".join(desc_en.split()[:50])

    # prepare outputs dict
    outputs = {"English": {"caption": cap_en, "description": desc_en}}

    # EN -> HI (Helsinki)
    if "Hindi" in sel:
        try:
            hi_tok = models["hi_tok"]; hi_mod = models["hi_model"]
            hi_in = hi_tok(cap_en, return_tensors="pt", padding=True).to(models["device"])
            hi_out = hi_mod.generate(**hi_in, max_new_tokens=200)
            cap_hi = hi_tok.decode(hi_out[0], skip_special_tokens=True)
            # description
            d_in = hi_tok(desc_en, return_tensors="pt", padding=True, truncation=True, max_length=512).to(models["device"])
            d_out = hi_mod.generate(**d_in, max_new_tokens=400)
            desc_hi = hi_tok.decode(d_out[0], skip_special_tokens=True)
        except Exception as e:
            cap_hi = f"Error: {e}"
            desc_hi = ""
        outputs["Hindi"] = {"caption": cap_hi, "description": desc_hi}
        st.subheader("Caption (Hindi)"); st.write(cap_hi)
        st.subheader("Description (Hindi)"); st.write(desc_hi)

    # EN -> TA (M2M100)
    if "Tamil" in sel:
        try:
            m2m_tok = models["m2m_tok"]; m2m_mod = models["m2m_model"]
            m2m_tok.src_lang = "en"
            tgt = "ta"
            forced = m2m_tok.get_lang_id(tgt)
            enc = m2m_tok(cap_en, return_tensors="pt").to(models["device"])
            gen = m2m_mod.generate(**enc, forced_bos_token_id=forced, max_new_tokens=128)
            cap_ta = m2m_tok.decode(gen[0], skip_special_tokens=True)
            # description
            encd = m2m_tok(desc_en, return_tensors="pt", truncation=True, max_length=1024).to(models["device"])
            gend = m2m_mod.generate(**encd, forced_bos_token_id=forced, max_new_tokens=400)
            desc_ta = m2m_tok.decode(gend[0], skip_special_tokens=True)
        except Exception as e:
            cap_ta = f"Error: {e}"
            desc_ta = ""
        outputs["Tamil"] = {"caption": cap_ta, "description": desc_ta}
        st.subheader("Caption (Tamil)"); st.write(cap_ta)
        st.subheader("Description (Tamil)"); st.write(desc_ta)

    # Bhojpuri: fallback plan: EN -> HI -> BHO via Bhojpuri LM paraphrase
    if "Bhojpuri" in sel:
        # If we have a Bhojpuri LM
        if models["bho_model"] and models["bho_tok"]:
            try:
                # first translate en -> hi (use results if present)
                base_hi = outputs.get("Hindi", {}).get("description", desc_en)
                # prompt the Bhojpuri LM: convert this Hindi sentence to Bhojpuri
                prompt = f"Convert this Hindi sentence to Bhojpuri:\n\n{base_hi}\n\nBhojpuri:"
                tok = models["bho_tok"]
                enc = tok(prompt, return_tensors="pt", truncation=True, max_length=512).to(models["device"])
                gen = models["bho_model"].generate(**enc, max_new_tokens=200, do_sample=False)
                bho_text = tok.decode(gen[0], skip_special_tokens=True)
                # split into caption/description roughly — keep as description
                outputs["Bhojpuri"] = {"caption": "(generated from Hindi)", "description": bho_text}
                st.subheader("Caption (Bhojpuri)"); st.write(outputs["Bhojpuri"]["caption"])
                st.subheader("Description (Bhojpuri)"); st.write(outputs["Bhojpuri"]["description"])
            except Exception as e:
                st.write("Bhojpuri generation error:", e)
                # fallback: show Hindi
                outputs["Bhojpuri"] = {"caption": outputs.get("Hindi", {}).get("caption", cap_en), "description": outputs.get("Hindi", {}).get("description", desc_en)}
                st.subheader("Caption (Bhojpuri - fallback Hindi)"); st.write(outputs["Bhojpuri"]["caption"])
                st.subheader("Description (Bhojpuri - fallback Hindi)"); st.write(outputs["Bhojpuri"]["description"])
        else:
            # no Bhojpuri LM available — fallback to Hindi output with note
            outputs["Bhojpuri"] = {"caption": outputs.get("Hindi", {}).get("caption", cap_en), "description": outputs.get("Hindi", {}).get("description", desc_en)}
            st.subheader("Caption (Bhojpuri - fallback Hindi)"); st.write(outputs["Bhojpuri"]["caption"])
            st.subheader("Description (Bhojpuri - fallback Hindi)"); st.write(outputs["Bhojpuri"]["description"])

    # download button
    out = []
    for lang, vals in outputs.items():
        out.append(f"Caption ({lang}): {vals.get('caption','')}\n\nDescription ({lang}): {vals.get('description','')}\n\n")
    st.download_button("Download all (txt)", data="".join(out), file_name="captions_multilang.txt")
