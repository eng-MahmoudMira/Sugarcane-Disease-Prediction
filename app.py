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

    # ---------- TABS ----------
    tab_detector, tab_about = st.tabs(["🔍 Detector", "ℹ️ About Project"])

    # ====== DETECTOR TAB ======
    with tab_detector:
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
                        "Try using clearer images or consulting an expert."
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

    # ====== ABOUT TAB ======
    with tab_about:
        st.title("ℹ️ About This Project")

        st.markdown("### 🌾 Problem Background")
        st.write(
            "Sugarcane is a vital crop for many economies, but leaf diseases can silently "
            "reduce yield, increase production costs, and threaten farmers' livelihoods. "
            "Traditional diagnosis relies on expert inspection, which is time-consuming "
            "and not always accessible for smallholder farmers."
        )

        st.markdown("### 🎯 Project Objective")
        st.write(
            "Our goal is to build an AI-based decision support tool that can classify sugarcane "
            "leaf images into different disease categories (or healthy) to support **early detection** "
            "and more informed crop management."
        )

        st.markdown("### 📊 Dataset & Classes")
        st.write(
            "We trained our model using a labeled image dataset of sugarcane leaves. "
            "Each image belongs to one of the following classes:"
        )
        st.markdown(
            "- **Healthy** – No clear signs of disease.\n"
            "- **Mosaic** – Patchy light/dark green patterns on the leaf.\n"
            "- **RedRot** – Serious fungal disease affecting stalk and leaves.\n"
            "- **Rust** – Rust-colored pustules on the leaf surface.\n"
            "- **Yellow** – General yellowing related to stress, nutrient deficiency, or early disease."
        )

        st.markdown("### 🧠 Model & Approach")
        st.write(
            "We used **transfer learning** with an EfficientNet-based architecture. "
            "The model was fine-tuned on our sugarcane dataset after applying preprocessing and "
            "augmentation such as resizing, normalization, and basic transformations. "
            "Our evaluation used metrics like accuracy and class probabilities to understand "
            "how confident the model is for each prediction."
        )

        st.markdown("### ⚙️ How to Use the App")
        st.markdown(
            "1. Capture or choose a clear image of a single sugarcane leaf.\n"
            "2. Avoid blurry images or very dark/over-exposed lighting.\n"
            "3. Upload the image in the **Detector** tab.\n"
            "4. Click **Predict** to see the predicted class and confidence.\n"
            "5. Read the **Disease Information & Advice** for basic guidance."
        )

        st.markdown("### ⚠️ Limitations")
        st.write(
            "- The model is trained on a specific dataset and may not generalize perfectly "
            "to all field conditions.\n"
            "- Overlapping leaves, extreme lighting, or very early-stage symptoms may reduce accuracy.\n"
            "- This tool should **support**, not replace, expert agricultural advice."
        )

        st.markdown("### 🚀 Future Work")
        st.write(
            "- Collect more diverse and real-world images from different regions.\n"
            "- Extend the model to handle more crop types and diseases.\n"
            "- Deploy as a mobile app for offline use in the field.\n"
            "- Integrate with a recommendation system for treatment plans."
        )

        st.markdown("---")
        st.markdown("### 👥 Team & Contacts")

        st.write(
            "You can add your names and LinkedIn profiles below. "
            "This section is great to show during your presentation or for anyone visiting the app."
        )

        st.markdown("**Team Members:**")
        st.markdown(
            """
            <table style="width:100%; table-layout: fixed; border:none; border-spacing:0;">
                <tr style="border:none;">
                    <td style="border:none;"><a href="https://www.linkedin.com/in/-mahmoudmira-/" target="_blank" style="text-decoration:none;"><img src="Members/Mahmoud_Mira.jpg" alt="Mahmoud Mira" style="max-width:100%; border:none; border-radius:85px; height:auto;"></td>
                    <td style="border:none;"><a href="https://www.linkedin.com/in/farah-bassiony-541930250/" target="_blank" style="text-decoration:none;"><img src="Members/Farah_Bassiony.jpeg" alt="Farah Bassiony" style="max-width:100%; border:none; border-radius:85px; height:auto;"></td>
                    <td style="border:none;"><a href="https://www.linkedin.com/in/yousef-shaban-003bba33b/" target="_blank" style="text-decoration:none;"><img src="https://media.licdn.com/dms/image/v2/D4D03AQGuEnIK8iXx2Q/profile-displayphoto-shrink_800_800/B4DZOOmWc3HUAg-/0/1733264264577?e=1766016000&v=beta&t=QwGSILAhnF93UAG_lR7_bynMa0QSwsiFJmYF0R1Cmzs" alt="Yousef Shaban" style="max-width:100%; border:none; border-radius:85px; height:auto;"></td>
                </tr>
                <tr style="border:none;">
                    <td style="text-align:center; border:none; padding:10px; margin: auto;">
                        <a href="https://www.linkedin.com/in/-mahmoudmira-/" target="_blank" style="text-decoration:none;">Mahmoud Mira</a>
                    </td>
                    <td style="text-align:center; border:none; padding:10px; margin: auto;">
                        <a href="https://www.linkedin.com/in/farah-bassiony-541930250/" target="_blank" style="text-decoration:none;">Farah Bassiony</a>
                    </td>
                    <td style="text-align:center; border:none; padding:10px; margin: auto;">
                        <a href="https://www.linkedin.com/in/yousef-shaban-003bba33b/" target="_blank" style="text-decoration:none;">Yousef Shaban</a>
                    </td>
                </tr>
            </table>
            """,
            unsafe_allow_html=True
        )

if __name__ == "__main__":
    main()



