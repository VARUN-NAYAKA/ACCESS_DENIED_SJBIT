from flask import Flask, request, jsonify
import tensorflow as tf
from PIL import Image
import io
import numpy as np

app = Flask(__name__)

# === Load TensorFlow Lite Model ===
interpreter = tf.lite.Interpreter(model_path="model/model.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# === Preprocess Image ===
def preprocess_image(image_bytes):
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert('RGB').resize((224, 224))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0).astype(np.float32)
        return img_array
    except Exception as e:
        raise ValueError(f"Image preprocessing failed: {e}")

# === Run Inference ===
def predict_with_tflite(image_array):
    interpreter.set_tensor(input_details[0]['index'], image_array)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    return output_data

# === API Endpoint ===
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'Missing image file'}), 400

    try:
        file = request.files['file'].read()
        img_array = preprocess_image(file)
        preds = predict_with_tflite(img_array)
        disease_label = int(np.argmax(preds))
        confidence = float(np.max(preds))

        # Placeholder bounding box (can be replaced with actual detection)
        bbox = [50, 50, 150, 150]

        # Kannada treatment tips (expand as needed)
        treatment_tip = {
            0: "ನೀಮ್ ಎಣ್ಣೆ ಸ್ಪ್ರೇ ಬಳಸಿ",
            1: "ಕಾಪರ್ ಫಂಗಿಸೈಡ್ ಅನ್ವಯಿಸಿ",
            2: "ಹಾನಿಗೊಳಗಾದ ಎಲೆಗಳನ್ನು ತೆಗೆದುಹಾಕಿ"
        }.get(disease_label, "ವೈಶಿಷ್ಟ್ಯಪೂರ್ಣ ಚಿಕಿತ್ಸೆ ಲಭ್ಯವಿಲ್ಲ")

        return jsonify({
            'disease_id': disease_label,
            'confidence': round(confidence, 3),
            'bbox': bbox,
            'treatment_tip': treatment_tip
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# === Run Server ===
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
