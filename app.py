from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import base64
import io
from PIL import Image

app = Flask(__name__)

# تحميل النموذج مرة واحدة عند تشغيل التطبيق
model = load_model("model.h5")


def preprocess_image(image):
    """
    تجهيز الصورة لتناسب نموذج MNIST
    """

    # تحويل إلى رمادي
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # عكس الألوان (MNIST: رقم أبيض وخلفية سوداء)
    gray = cv2.bitwise_not(gray)

    # إزالة الضوضاء
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    # Threshold
    _, thresh = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # إيجاد حدود الرقم
    contours, _ = cv2.findContours(
        thresh,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if len(contours) == 0:
        return None, None

    contour = max(contours, key=cv2.contourArea)

    x, y, w, h = cv2.boundingRect(contour)

    digit = thresh[y:y+h, x:x+w]

    size = max(w, h) + 20

    square = np.zeros((size, size), dtype=np.uint8)

    x_offset = (size - w) // 2
    y_offset = (size - h) // 2

    square[
        y_offset:y_offset+h,
        x_offset:x_offset+w
    ] = digit

    final = cv2.resize(square, (28, 28))

    preview = cv2.imencode(".png", final)[1]
    preview = base64.b64encode(preview).decode()

    final = final.astype("float32") / 255.0

    final = final.reshape(1, 28, 28, 1)

    return final, preview


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict_api", methods=["POST"])
def predict():

    try:

        data = request.get_json()

        image_data = data["image_data"]

        encoded = image_data.split(",")[1]

        image_bytes = base64.b64decode(encoded)

        image = Image.open(io.BytesIO(image_bytes))

        image = np.array(image)

        if image.shape[2] == 4:
            image = image[:, :, :3]

        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        processed, preview = preprocess_image(image)

        if processed is None:

            return jsonify({

                "success": False,

                "error": "لم يتم العثور على رقم."

            })

        prediction = model.predict(processed, verbose=0)

        digit = int(np.argmax(prediction))

        confidence = float(np.max(prediction) * 100)

        probabilities = prediction[0].tolist()

        return jsonify({

            "success": True,

            "digit": digit,

            "confidence": round(confidence, 2),

            "probabilities": probabilities,

            "processed_image": preview

        })

    except Exception as e:

        return jsonify({

            "success": False,

            "error": str(e)

        })


if __name__ == "__main__":

    app.run(
        debug=True,
        host="0.0.0.0",
        port=5000
    )