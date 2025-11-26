import streamlit as st
import numpy as np
from PIL import Image
from tensorflow import keras
import pandas as pd

# --------- CONFIG ----------
CLASS_NAMES = ["Healthy", "Mosaic", "RedRot", "Rust", "Yellow"]
IMG_SIZE = (224, 224)   # same as training
MODEL_PATH = "efficientnet_transfer.h5"
# ----------------------------

# Extra info to show helpful context
DISEASE_INFO = {
    "Healthy": {
        "description": "The leaf appears healthy with no clear visual signs of disease.",
        "advice": "Maintain current farming practices and keep monitoring the crop regularly."
    },
    "Mosaic": {
        "description": "Mosaic disease often causes patchy, light and dark green patterns on leaves.",
        "advice": "Isolate suspicious plants, monitor spread, and consult an agronomist for virus management."
    },
    "RedRot": {
        "description": "Red rot is a serious fungal disease that can significantly reduce yield.",
        "advice": "Remove and destroy infected plants if possible and consider resistant varieties in future seasons."
    },
    "Rust": {
        "description": "Rust appears as powdery rust-colored pustules on the leaf surface.",
        "advice": "Avoid overhead irrigation, and seek advice on appropriate fungicide use if the infection is severe."
    },
    "Yellow": {
        "description": "Yellowing may indicate nutrient deficiency, stress, or early-stage disease.",
        "advice": "Check soil nutrients, irrigation, and monitor leaves for progression or other symptoms."
    }
}


@st.cache_resource
def load_sugarcane_model():
    model = keras.models.load_model(MODEL_PATH)
    return model


def preprocess_image(img: Image.Image) -> np.ndarray:

    # Resize
    img = img.resize(IMG_SIZE)

    # Convert to array
    img_array = np.array(img)

    # Ensure 3 channels (RGB)
    if len(img_array.shape) == 2:          # grayscale
        img_array = np.stack([img_array] * 3, axis=-1)
    elif img_array.shape[-1] == 4:         # RGBA -> RGB
        img_array = img_array[..., :3]

    img_array = img_array.astype("float32")

    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)   # shape: (1, 224, 224, 3)
    return img_array


def predict_image(model, img: Image.Image):
    processed = preprocess_image(img)
    preds = model.predict(processed)[0]  # shape: (num_classes,)
    predicted_index = int(np.argmax(preds))
    predicted_class = CLASS_NAMES[predicted_index]
    confidence = float(preds[predicted_index])
    return predicted_class, confidence, preds


def main():
    st.set_page_config(
        page_title="Sugarcane Leaf Disease Detection",
        page_icon="🌱",
        layout="centered",
    )

    # ---------- SIDEBAR ----------
    with st.sidebar:
        st.title("🌱 CaneGuard AI")
        st.markdown(
            "An AI-powered tool to help detect **sugarcane leaf diseases** from images."
        )
        st.markdown("### How to use:")
        st.markdown(
            "1. Take a clear photo of a sugarcane leaf.\n"
            "2. Upload the image using the uploader.\n"
            "3. Click **Predict** to see the result.\n"
            "4. Read the **disease info & advice** section."
        )
        st.markdown("### Model Info")
        st.markdown(
            "- Architecture: EfficientNet (transfer learning)\n"
            "- Input size: 224×224\n"
            "- Classes: 5 (Healthy, Mosaic, RedRot, Rust, Yellow)"
        )

    # ---------- MAIN AREA ----------
    st.title("🌱 Sugarcane Leaf Disease Prediction")
    st.write(
        "Upload a sugarcane leaf image and let our AI model predict the disease class."
    )

    # Load model once
    with st.spinner("Loading AI model..."):
        model = load_sugarcane_model()

    uploaded_file = st.file_uploader(
        "Upload a sugarcane leaf image (JPG/PNG)...", type=["jpg", "jpeg", "png", "webp"]
    )

    if uploaded_file is not None:
        img = Image.open(uploaded_file)
        st.image(img, caption="Uploaded Image", use_container_width=True)

        if st.button("🔍 Predict"):
            with st.spinner("Analyzing..."):
                predicted_class, confidence, all_probs = predict_image(model, img)

            st.subheader("Prediction")
            st.write(f"**Class:** {predicted_class}")
            st.write(f"**Confidence:** {confidence * 100:.2f}%")

            # Confidence warning if model is not very sure
            if confidence < 0.6:
                st.warning(
                    "The model is not very confident about this prediction. "
                    "Consider using more images or consulting an expert."
                )

            # Disease info & advice
            st.subheader("Disease Information & Advice")
            info = DISEASE_INFO.get(predicted_class, None)
            if info:
                st.markdown(f"**Description:** {info['description']}")
                st.markdown(f"**Suggested Action:** {info['advice']}")
            else:
                st.write("No extra information available for this class yet.")

            # Probabilities as a table + chart
            st.subheader("Class Probabilities")
            prob_dict = {cls: float(p) for cls, p in zip(CLASS_NAMES, all_probs)}
            df_probs = pd.DataFrame(
                {"Class": list(prob_dict.keys()), "Probability": list(prob_dict.values())}
            ).set_index("Class")

            st.dataframe(df_probs.style.format({"Probability": "{:.2%}"}))
            st.bar_chart(df_probs)

            st.success("You can try another image if you like!")

    st.markdown("---")
    st.caption("SIC Graduation Project – Sugarcane Leaf Disease Prediction")


if __name__ == "__main__":
    main()
