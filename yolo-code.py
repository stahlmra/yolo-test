import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2

st.set_page_config(page_title="YOLO Kleidungserkennung")

st.title("👕 YOLO Kleidungserkennung")
st.write("Lade ein Bild hoch – das Modell erkennt Kleidung, Schuhe, Taschen und Accessoires.")

# Modell laden (nur einmal)
@st.cache_resource
def load_model():
    model = YOLO("kesimeg/yolov8n-clothing-detection")
    return model

model = load_model()

# Sidebar Einstellungen
st.sidebar.header("Einstellungen")
conf = st.sidebar.slider("Confidence", 0.1, 0.9, 0.35, 0.05)
iou = st.sidebar.slider("IOU", 0.1, 0.9, 0.45, 0.05)

# Upload
uploaded = st.file_uploader("Bild hochladen", type=["jpg","jpeg","png"])

if uploaded:
    image = Image.open(uploaded).convert("RGB")
    st.image(image, caption="Originalbild", use_column_width=True)

    if st.button("🔍 Kleidung erkennen"):

        with st.spinner("Analysiere Bild..."):

            results = model.predict(
                source=np.array(image),
                conf=conf,
                iou=iou,
                verbose=False
            )[0]

            # Ergebnisbild
            result_img = results.plot()
            result_img = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)

            st.image(result_img, caption="Ergebnis", use_column_width=True)

            # Klassen anzeigen
            if len(results.boxes) > 0:

                st.subheader("Erkannte Objekte")

                for box in results.boxes:
                    cls = int(box.cls)
                    label = results.names[cls]
                    score = float(box.conf)

                    st.write(f"• **{label}** ({score:.2%})")

            else:
                st.warning("Keine Objekte erkannt.")
