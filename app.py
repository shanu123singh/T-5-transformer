import streamlit as st

from transformers import (
    T5Tokenizer,
    T5ForConditionalGeneration
)

# -----------------------------
# Load Trained Model
# -----------------------------

@st.cache_resource
def load_soil_model():

    tokenizer = T5Tokenizer.from_pretrained(
            "soil_model"
    )

    model = T5ForConditionalGeneration.from_pretrained(
        "soil_model"
    )

    return tokenizer, model


tokenizer, model = load_soil_model()

# -----------------------------
# Prediction Function
# -----------------------------

def predict_crop(
    N,
    P,
    K,
    temperature,
    humidity,
    ph,
    rainfall
):

    input_text = f"""
soil_nitrogen: {N}
soil_phosphorus: {P}
soil_potassium: {K}
temperature: {temperature}
humidity: {humidity}
ph: {ph}
rainfall: {rainfall}

question: which crop should i grow?
"""

    inputs = tokenizer(
        input_text,
        return_tensors="pt",
        truncation=True,
        max_length=64
    )

    outputs = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_length=20,
        num_beams=5
    )

    prediction = tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    )

    return prediction


# -----------------------------
# Streamlit UI
# -----------------------------

st.set_page_config(
    page_title="AI Soil Advisor",
    page_icon="🌱"
)

st.title("🌱 AI Soil Advisor")
st.write("Crop Recommendation using T5 Transformer")

st.subheader("Enter Soil Information")

N = st.number_input(
    "Nitrogen (N)",
    min_value=0.0,
    value=90.0
)

P = st.number_input(
    "Phosphorus (P)",
    min_value=0.0,
    value=42.0
)

K = st.number_input(
    "Potassium (K)",
    min_value=0.0,
    value=43.0
)

temperature = st.number_input(
    "Temperature",
    value=21.0
)

humidity = st.number_input(
    "Humidity",
    value=82.0
)

ph = st.number_input(
    "pH",
    value=6.5
)

rainfall = st.number_input(
    "Rainfall",
    value=203.0
)

# -----------------------------
# Predict Button
# -----------------------------

if st.button("Recommend Crop"):

    with st.spinner("Analyzing Soil..."):

        crop = predict_crop(
            N,
            P,
            K,
            temperature,
            humidity,
            ph,
            rainfall
        )

    st.success(
        f"Recommended Crop: {crop}"
    )

    st.info(
        f"{crop} is predicted as the most suitable crop for the given soil conditions."
    )