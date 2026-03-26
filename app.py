from flask import Flask, render_template, request
import cv2
import numpy as np
from tensorflow import keras
import base64

app = Flask(__name__)

# ---------------------------
# LOAD MODEL
# ---------------------------
try:
    model = keras.models.load_model("emotion_model.h5")
    print("✅ Model loaded successfully")
except Exception as e:
    print("❌ Model loading failed:", e)
    model = None

# ---------------------------
# LOAD FACE CASCADE
# ---------------------------
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

if face_cascade.empty():
    print("❌ Haar Cascade not loaded")
else:
    print("✅ Haar Cascade loaded")

# ---------------------------
# LABELS & MAPPING
# ---------------------------
emotion_labels_full = [
    "Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"
]

emotion_music = {
    'Happy': 'Pop Songs',
    'Sad': 'Melody Songs',
    'Angry': 'Rock Songs',
    'Neutral': 'Instrumental',
    'Fear': 'Devotional - Calm',
    'Surprise': 'EDM'
}

# ---------------------------
# ROUTES
# ---------------------------
@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    print("🔥 /predict route hit")

    # Check model
    if model is None:
        return render_template("result.html",
                               emotion="Error",
                               music="Model not loaded",
                               confidence=0)

    image_data = request.form.get("image")

    if not image_data:
        return render_template("result.html",
                               emotion="Error",
                               music="No image received",
                               confidence=0)

    try:
        # ---------------------------
        # SAFE BASE64 DECODE
        # ---------------------------
        if "," in image_data:
            image_data = image_data.split(",")[1]
        else:
            return render_template("result.html",
                                   emotion="Error",
                                   music="Invalid image format",
                                   confidence=0)

        image_bytes = base64.b64decode(image_data)

        np_arr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if img is None:
            raise ValueError("Image decoding failed")

        # ---------------------------
        # PREPROCESSING
        # ---------------------------
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)

        gray = cv2.GaussianBlur(gray, (3, 3), 0)

        # ---------------------------
        # FACE DETECTION
        # ---------------------------
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=4,
            minSize=(60, 60)
        )

        print("Faces detected:", len(faces))

        if len(faces) == 0:
            return render_template("result.html",
                                   emotion="No Face Detected",
                                   music="Face camera properly with good lighting",
                                   confidence=0)

        # Take largest face
        faces = sorted(faces, key=lambda x: x[2]*x[3], reverse=True)
        x, y, w, h = faces[0]
        face = gray[y:y+h, x:x+w]

        # ---------------------------
        # MODEL INPUT PREP
        # ---------------------------
        face = cv2.resize(face, (48, 48))
        face = face / 255.0
        face = face.reshape(1, 48, 48, 1)

        # ---------------------------
        # PREDICTION
        # ---------------------------
        prediction = model.predict(face, verbose=0)
        confidence = float(np.max(prediction))
        emotion = emotion_labels_full[np.argmax(prediction)]

        # Handle weak predictions
        if emotion == "Disgust" or confidence < 0.60:
            emotion = "Neutral"

        music = emotion_music.get(emotion, "Instrumental")

        print(f"Prediction: {emotion} ({confidence:.2f})")

        return render_template("result.html",
                               emotion=emotion,
                               music=music,
                               confidence=round(confidence * 100, 2))

    except Exception as e:
        print("❌ ERROR:", e)
        return render_template("result.html",
                               emotion="Error",
                               music="Processing failed",
                               confidence=0)


# ---------------------------
# RUN APP
# ---------------------------
if __name__ == "__main__":
    app.run(debug=True)