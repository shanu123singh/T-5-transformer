import streamlit as st
import os
from transformers import T5Tokenizer, T5ForConditionalGeneration

st.set_page_config(
    page_title="AI Soil Advisor",
    page_icon="🌱"
)

# -----------------------------
# LOAD MODEL
# -----------------------------

@st.cache_resource
def load_soil_model():

    base_dir = os.path.dirname(os.path.abspath(__file__))
    MODEL_PATH = os.path.join(base_dir, "soil_model_v2")

    try:
        if os.path.exists(MODEL_PATH) and len(os.listdir(MODEL_PATH)) > 0:

            st.sidebar.success("✅ Loading Trained Model (soil_model_v2)")

            tokenizer = T5Tokenizer.from_pretrained(MODEL_PATH)
            model = T5ForConditionalGeneration.from_pretrained(MODEL_PATH)

        else:

            st.sidebar.warning("⚠️ soil_model_v2 not found. Loading t5-small")

            tokenizer = T5Tokenizer.from_pretrained("t5-small")
            model = T5ForConditionalGeneration.from_pretrained("t5-small")

        return tokenizer, model

    except Exception as e:
        st.error(f"Model Loading Error: {e}")
        st.stop()


tokenizer, model = load_soil_model()

# -----------------------------
# PREDICTION FUNCTION (FIXED)
# -----------------------------

def predict_crop(N, P, K, temperature, humidity, ph, rainfall):

    # CLEAN PROMPT (IMPORTANT FIX)
    input_text = f"""
You are an agricultural expert.
Predict only ONE crop name based on soil data.

N:{N}
P:{P}
K:{K}
Temperature:{temperature}
Humidity:{humidity}
pH:{ph}
Rainfall:{rainfall}

Answer:
"""

    inputs = tokenizer(
        input_text,
        return_tensors="pt",
        truncation=True,
        max_length=128
    )

    outputs = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],

        max_length=5,
        num_beams=10,
        early_stopping=True,
        repetition_penalty=2.5,
        no_repeat_ngram_size=2
    )

    prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)

    prediction = prediction.strip().lower()

    # REMOVE GARBAGE OUTPUT
    if "n:" in prediction or "ph:" in prediction or "rainfall:" in prediction:
        return "⚠️ model needs retraining"

    if prediction == "":
        return "⚠️ no prediction"

    return prediction


# -----------------------------
# UI
# -----------------------------

st.title("🌱 AI Soil Advisor")
st.write("Crop Recommendation using T5 Transformer")

N = st.number_input("Nitrogen (N)", 0.0, value=90.0)
P = st.number_input("Phosphorus (P)", 0.0, value=42.0)
K = st.number_input("Potassium (K)", 0.0, value=43.0)

temperature = st.number_input("Temperature", value=21.0)
humidity = st.number_input("Humidity", value=82.0)
ph = st.number_input("pH", value=6.5)
rainfall = st.number_input("Rainfall", value=203.0)

if st.button("🌾 Recommend Crop"):

    with st.spinner("Analyzing soil..."):

        result = predict_crop(N, P, K, temperature, humidity, ph, rainfall)

    st.success(f"🌾 Recommended Crop: {result}")