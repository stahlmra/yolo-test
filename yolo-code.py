import streamlit as st
from ultralyticsplus import YOLO
from PIL import Image

@st.cache_resource
def load_model():
    return YOLO("kesimeg/yolov8n-clothing-detection")

model = load_model()

st.title("Kleidung erkennen")

bild = st.file_uploader("Foto hochladen", type=["jpg", "jpeg", "png"])

if bild:
    img = Image.open(bild)
    st.image(img)

    if st.button("Analysieren"):
        with st.spinner("..."):
            results = model.predict(img, conf=0.25)

            st.image(results[0].plot(), caption="Ergebnis")

            if len(results[0].boxes) == 0:
                st.write("Nichts erkannt.")
            else:
                st.write("Gefunden:")
                for box in results[0].boxes:
                    label = results[0].names[int(box.cls)]
                    conf = box.conf.item()
                    st.write(f"• {label} ({conf:.0%})")import streamlit as st
from ultralytics import YOLO
from PIL import Image

# Modell laden (cached, lädt nur einmal)
@st.cache_resource
def get_model():
    return YOLO("kesimeg/yolov8n-clothing-detection")

model = get_model()

st.title("Kleidung erkennen")

bild = st.file_uploader("Foto hochladen", type=["jpg", "jpeg", "png"])

if bild:
    img = Image.open(bild)
    st.image(img, caption="Dein Bild")

    if st.button("Jetzt analysieren"):
        with st.spinner("Läuft..."):
            ergebnisse = model(img)

            # Zeige Bild mit Boxen
            annotiert = ergebnisse[0].plot()           # numpy-Array mit Zeichnungen
            st.image(annotiert, caption="Erkennung")

            # Liste gefundene Dinge
            if len(ergebnisse[0].boxes) > 0:
                st.write("Gefunden:")
                for box in ergebnisse[0].boxes:
                    name = ergebnisse[0].names[int(box.cls)]
                    sicherheit = float(box.conf)
                    st.write(f"- {name} ({sicherheit:.0%})")
            else:
                st.write("Nichts erkannt.")
