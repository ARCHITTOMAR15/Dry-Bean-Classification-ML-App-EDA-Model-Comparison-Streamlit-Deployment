import streamlit as st
import numpy as np
import pandas as pd
import joblib
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(page_title="🌱 Bean Classifier", layout="wide")

# =========================
# THEME TOGGLE
# =========================
theme = st.sidebar.radio("🌗 Theme", ["Light Mode", "Dark Mode"])

# =========================
# COLORS
# =========================
if theme == "Dark Mode":
    bg = "#0f172a"
    card_bg = "rgba(255,255,255,0.05)"
    text = "#ffffff"
else:
    bg = "linear-gradient(-45deg, #e6f4ea, #d0f0fd, #fdfcdc, #e6f4ea)"
    card_bg = "rgba(255,255,255,0.6)"
    text = "#000000"

# =========================
# PREMIUM CSS
# =========================
st.markdown(f"""
<style>

/* Animated Background */
.stApp {{
    background: {bg};
    background-size: 400% 400%;
    animation: gradientMove 12s ease infinite;
    color: {text};
}}

/* Metric Tiles */
.metric-tile {{
    padding: 15px;
    border-radius: 15px;
    margin-bottom: 10px;
    color: white;
    text-align: center;
    font-weight: bold;
    box-shadow: 0 8px 20px rgba(0,0,0,0.2);
    transition: 0.3s;
}}

.metric-tile:hover {{
    transform: scale(1.05);
    box-shadow: 0 0 20px rgba(82,183,136,0.7);
}}

/* TOP LEFT BRAND TILE */
.brand-tile {{
    background: linear-gradient(135deg, #ff9f1c, #ff6b6b, #845ef7);
    padding: 10px 18px;
    border-radius: 12px;
    color: white;
    font-weight: bold;
    display: inline-block;
    box-shadow: 0 6px 15px rgba(0,0,0,0.2);
    animation: fadeIn 1s ease;
}}

@keyframes gradientMove {{
    0% {{background-position: 0% 50%;}}
    50% {{background-position: 100% 50%;}}
    100% {{background-position: 0% 50%;}}
}}

.main-title {{
    text-align: center;
    font-size: 48px;
    font-weight: 800;
    background: linear-gradient(90deg, #2E8B57, #52b788, #74c69d);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}}

.card {{
    background: {card_bg};
    backdrop-filter: blur(15px);
    border-radius: 20px;
    padding: 25px;
    margin-bottom: 20px;
}}

.result-box {{
    text-align: center;
    font-size: 34px;
    font-weight: bold;
    color: white;
    padding: 20px;
    border-radius: 15px;
    background: linear-gradient(135deg, #20c997, #38d9a9, #63e6be);
    box-shadow: 0 10px 30px rgba(0,0,0,0.3);
    animation: fadeIn 0.8s ease;
}}

.stButton>button {{
    background: linear-gradient(135deg, #ff6b6b, #f06595, #845ef7);
    color: white;
    border-radius: 16px;
    height: 3.5em;
    font-size: 20px;
    font-weight: bold;
    letter-spacing: 1px;
    transition: all 0.3s ease;
    box-shadow: 0 8px 25px rgba(0,0,0,0.3);
}}

.stButton>button:hover {{
    transform: scale(1.12);
    box-shadow: 0 0 25px rgba(132,94,247,0.8);
}}

/* DOWNLOAD BUTTON (ADDED) */
.stDownloadButton>button {{
    background: linear-gradient(135deg, #ff9f1c, #ff6b6b, #845ef7);
    color: white;
    border-radius: 16px;
    height: 3.2em;
    font-size: 18px;
    font-weight: bold;
    transition: all 0.3s ease;
    box-shadow: 0 8px 25px rgba(0,0,0,0.3);
}}

.stDownloadButton>button:hover {{
    transform: scale(1.1);
    box-shadow: 0 0 25px rgba(255,107,107,0.8);
}}

@keyframes fadeIn {{
    from {{opacity:0;}}
    to {{opacity:1;}}
}}

</style>
""", unsafe_allow_html=True)

# =========================
# TOP LEFT BRAND TILE
# =========================
st.markdown('<div class="brand-tile">Created by Archit Tomar</div>', unsafe_allow_html=True)

# =========================
# LOAD FILES
# =========================
model = joblib.load("best_model_catboost.pkl")
scaler = joblib.load("scaler.pkl")
label_encoder = joblib.load("label_encoder.pkl")

# =========================
# FEATURES
# =========================
log_features = [
    'Area','Perimeter','MinorAxisLength','AspectRation',
    'Eccentricity','ConvexArea','EquivDiameter',
    'Extent','Solidity','roundness','ShapeFactor4'
]

model_features = [
    'Area','Perimeter','MajorAxisLength','MinorAxisLength',
    'AspectRation','Eccentricity','ConvexArea','EquivDiameter',
    'Extent','Solidity','roundness','Compactness',
    'ShapeFactor1','ShapeFactor2','ShapeFactor3','ShapeFactor4'
]

# =========================
# MEANS
# =========================
feature_means = {
    'Area':50000,'Perimeter':800,'MajorAxisLength':300,
    'MinorAxisLength':200,'AspectRation':1.5,'Eccentricity':0.8,
    'ConvexArea':52000,'EquivDiameter':250,'Extent':0.7,
    'Solidity':0.95,'roundness':0.85,'Compactness':0.9,
    'ShapeFactor1':0.005,'ShapeFactor2':0.001,
    'ShapeFactor3':0.8,'ShapeFactor4':0.98
}

# =========================
# PREPROCESS
# =========================
def preprocess(df):
    df = df.copy()
    for col in log_features:
        if col in df.columns:
            df[col] = np.log1p(df[col])
    return scaler.transform(df)

# =========================
# SIDEBAR NAVIGATION
# =========================
st.sidebar.markdown("### 🔍 Navigation")
mode = st.sidebar.radio("Mode", ["Single Prediction", "Batch Prediction"])

st.sidebar.markdown(f"""
<div class="metric-tile" style="background: linear-gradient(135deg, #4361ee, #4895ef);">
Mode<br>{mode}
</div>
""", unsafe_allow_html=True)

# =========================
# SIDEBAR MODEL INFO
# =========================
st.sidebar.markdown("### ⚙️ Model Info")

st.sidebar.markdown('<div class="metric-tile" style="background: linear-gradient(135deg,#ff6b6b,#ff8787);">Model<br>CatBoost</div>', unsafe_allow_html=True)
st.sidebar.markdown('<div class="metric-tile" style="background: linear-gradient(135deg,#f59f00,#ffd43b);">Train Accuracy<br>0.9481</div>', unsafe_allow_html=True)
st.sidebar.markdown('<div class="metric-tile" style="background: linear-gradient(135deg,#20c997,#63e6be);">Test Accuracy<br>0.9189</div>', unsafe_allow_html=True)
st.sidebar.markdown('<div class="metric-tile" style="background: linear-gradient(135deg,#845ef7,#b197fc);">Weighted Avg F1 Score<br>0.92</div>', unsafe_allow_html=True)
st.sidebar.markdown('<div class="metric-tile" style="background: linear-gradient(135deg,#0f2027,#2c5364,#00c6ff);">Data Point Used In Model<br>13000(Approx)</div>', unsafe_allow_html=True)





# =========================
# TITLE
# =========================
st.markdown('<div class="main-title">🌱 Dry Bean Classification</div>', unsafe_allow_html=True)

# =========================
# SINGLE MODE
# =========================
if mode == "Single Prediction":

    st.markdown('<div class="card">', unsafe_allow_html=True)

    cols = st.columns(4)
    input_data = {}

    for i, feature in enumerate(model_features):
        with cols[i % 4]:
            val = float(feature_means[feature])
            input_data[feature] = st.slider(feature, 0.0, val*2, val)

    st.markdown('</div>', unsafe_allow_html=True)

    df = pd.DataFrame([input_data])

    col1, col2 = st.columns(2)
    prediction_label = None

    with col1:
        if st.button("🚀 Predict"):
            processed = preprocess(df)
            pred = model.predict(processed)
            prediction_label = label_encoder.inverse_transform(pred)[0]
            st.markdown(f'<div class="result-box">🌱 Bean Type : {prediction_label}</div>', unsafe_allow_html=True)

    def create_pdf(prediction, inputs):
        doc = SimpleDocTemplate("report.pdf")
        styles = getSampleStyleSheet()
        content = []

        content.append(Paragraph("Bean Classification Report", styles["Title"]))
        content.append(Spacer(1, 20))

        table_data = [["Feature", "Value"]]
        for k, v in inputs.items():
            table_data.append([k, str(v)])

        table_data.append(["Predicted Bean Type", prediction])

        table = Table(table_data)
        table.setStyle(TableStyle([
            ('BACKGROUND',(0,0),(-1,0),colors.grey),
            ('TEXTCOLOR',(0,0),(-1,0),colors.white),
            ('GRID',(0,0),(-1,-1),1,colors.black),
        ]))

        content.append(table)
        doc.build(content)

    with col2:
        if prediction_label:
            create_pdf(prediction_label, input_data)
            with open("report.pdf", "rb") as f:
                st.download_button("📄 Download Report", f, file_name="report.pdf")

# =========================
# BATCH MODE
# =========================
else:
    file = st.file_uploader("Upload CSV", type=["csv"])

    if file:
        df = pd.read_csv(file)
        st.dataframe(df.head())

        processed = preprocess(df)
        preds = model.predict(processed)
        df["Prediction"] = label_encoder.inverse_transform(preds)

        st.dataframe(df)
        st.download_button("⬇️ Download", df.to_csv(index=False), "predictions.csv")
