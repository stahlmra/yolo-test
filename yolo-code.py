import streamlit as st
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
