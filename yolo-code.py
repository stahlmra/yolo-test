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
                    st.write(f"• {label} ({conf:.0%})")
