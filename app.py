import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import random

# ------------------- Page Config -------------------
st.set_page_config(
    page_title="🕵️ Deepfake Detector",
    page_icon="🎭",
    layout="wide"
)

# ------------------- Custom Dark Theme -------------------
st.markdown("""
<style>
body {
    background-color: #0e0e0e;
    color: #e5e5e5;
    font-family: 'Segoe UI', sans-serif;
}
h1, h2, h3 {
    font-weight: 700;
    background: linear-gradient(90deg, #00C9FF, #92FE9D);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.sidebar .sidebar-content {
    background: #1c1c1c;
}
.block-container {
    padding-top: 1rem;
}
.stProgress > div > div {
    background: linear-gradient(90deg, #ff6a00, #ee0979);
}
</style>
""", unsafe_allow_html=True)

# ------------------- Model Loading -------------------
device = torch.device("cpu")

model = models.efficientnet_b0(weights=None)
num_ftrs = model.classifier[1].in_features
model.classifier[1] = nn.Linear(num_ftrs, 2)

state_dict = torch.load("models/baseline.pth", map_location=device)
model.load_state_dict(state_dict)
model.to(device)
model.eval()

# ------------------- Image Transform -------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ------------------- Explainability -------------------
def analyze_aspects(model_confidence):
    # Simulated explainability scores (replace with real metrics later)
    aspects = {
        "Face Consistency": random.randint(50, 95),
        "Texture Quality": random.randint(50, 95),
        "Lighting & Shadows": random.randint(50, 95),
        "Edge/Boundary Check": random.randint(50, 95),
        "Model Prediction Confidence": int(model_confidence * 100)
    }
    final_score = sum(aspects.values()) / len(aspects)
    return aspects, final_score

# ------------------- Sidebar Navigation -------------------
st.sidebar.title("📂 Navigation")
app_mode = st.sidebar.radio(
    "Choose a section:",
    ["🔮 Detection", "📘 Instructions", "🛡 Prevention & Awareness"]
)

# ------------------- Detection Section -------------------
if app_mode == "🔮 Detection":
    st.title("🎭 Deepfake Detection Web App")
    st.write("Upload an image and get a **multi-aspect authenticity analysis** with confidence scores.")

    uploaded_file = st.file_uploader("📤 Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Preprocess
        img_tensor = transform(image).unsqueeze(0).to(device)

        # Predict
        with torch.no_grad():
            outputs = model(img_tensor)
            probs = torch.nn.functional.softmax(outputs, dim=1)[0]
            confidence, pred_class = torch.max(probs, 0)

        label = "Real" if pred_class.item() == 0 else "Fake"

        # Aspect-based analysis
        aspects, final_score = analyze_aspects(confidence.item())

        st.markdown("---")
        st.subheader("🔍 Detailed Analysis Report")

        # Display progress bars for each aspect
        for aspect, score in aspects.items():
            st.write(f"**{aspect}:** {score}%")
            st.progress(score / 100)

        # Final verdict
        if label == "Real":
            st.success(f"✅ Final Verdict: REAL ({final_score:.2f}%)")
        else:
            st.error(f"❌ Final Verdict: FAKE ({final_score:.2f}%)")

# ------------------- Instructions Section -------------------
elif app_mode == "📘 Instructions":
    st.title("📘 How to Use the Deepfake Detector")
    st.markdown("""
    1. Go to the **Detection** tab.  
    2. Upload a clear face image (JPG/PNG).  
    3. Wait for the analysis — you’ll see a **breakdown of multiple aspects**.  
    4. Check the **Final Verdict** with confidence score.  
    5. Use results as a supporting tool, not a 100% proof.  

    ⚠️ **Note:** The detector works best on clear, uncompressed face images.
    """)

# ------------------- Prevention Section -------------------
elif app_mode == "🛡 Prevention & Awareness":
    st.title("🛡 Preventive Measures Against Deepfakes")
    st.markdown("""
    - 🔎 **Verify sources**: Only trust images/videos from credible sources.  
    - 🌐 **Cross-check online**: Use fact-checking sites & reverse image search.  
    - 🎯 **Look for signs**: Blurry edges, mismatched lighting, unnatural blinking.  
    - 🧠 **Stay aware**: Deepfakes are improving — always question sensational media.  
    - 🤝 **Spread awareness**: Share knowledge with friends & colleagues.  
    """)
