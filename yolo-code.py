import streamlit as st
from ultralytics import YOLO
from PIL import Image
import cv2
import numpy as np
import io

# ────────────────────────────────────────────────
# Modell laden (wird beim ersten Start automatisch heruntergeladen)
@st.cache_resource
def load_model():
    return YOLO("kesimeg/yolov8n-clothing-detection")

model = load_model()

# ────────────────────────────────────────────────
st.title("Kleidungserkennung mit YOLO")
st.write("Lade ein Foto hoch – das Modell erkennt Kleidung, Schuhe, Taschen & Accessoires.")

# Optionen in der Sidebar
st.sidebar.header("Einstellungen")
conf_threshold = st.sidebar.slider("Confidence Threshold", 0.1, 0.9, 0.35, 0.05)
iou_threshold  = st.sidebar.slider("IOU Threshold (NMS)", 0.1, 0.9, 0.45, 0.05)

# Bild-Upload
uploaded_file = st.file_uploader("Bild hochladen (jpg, png, ...)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Bild laden
    image = Image.open(uploaded_file)
    st.image(image, caption="Originalbild", use_column_width=True)

    # Predict-Button
    if st.button("Kleidung erkennen"):
        with st.spinner("Analysiere Bild ..."):
            # Ultralytics Predict
            results = model.predict(
                source=image,
                conf=conf_threshold,
                iou=iou_threshold,
                verbose=False
            )[0]  # erstes Ergebnis (ein Bild)

            # Annotiertes Bild erzeugen
            annotated_img = results.plot()  # numpy array mit Boxen + Labels

            # Farben von BGR → RGB für PIL/Streamlit
            annotated_img_rgb = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
            annotated_pil = Image.fromarray(annotated_img_rgb)

            # Ergebnis anzeigen
            st.image(annotated_pil, caption="Erkennungsergebnis", use_column_width=True)

            # Gefundene Objekte auflisten
            if len(results.boxes) > 0:
                st.subheader("Erkannte Objekte:")
                for box in results.boxes:
                    cls_id = int(box.cls)
                    label = results.names[cls_id]
                    conf = float(box.conf)
                    st.write(f"• {label} – Sicherheit: {conf:.1%}")
            else:
                st.info("Keine Kleidung/Taschen/Schuhe erkannt (Confidence zu hoch?)")

# ────────────────────────────────────────────────
st.markdown("---")
st.caption(
    "Modell: [kesimeg/yolov8n-clothing-detection](https://huggingface.co/kesimeg/yolov8n-clothing-detection)  •  "
    "Framework: Ultralytics YOLO • 4 Klassen (Clothing, Shoes, Bags, Accessories)"
)
