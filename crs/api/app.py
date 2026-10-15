from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import numpy as np

app = Flask(__name__)
model = load_model('mobileNetV2_model.h5')
class_names = ['Angry', 'Fear', 'Happy', 'Sad']

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    image_file = request.files['image']
    # تأكد أن الصورة RGB
    image = Image.open(image_file).convert('RGB').resize((224, 224))
    img_array = img_to_array(image)
    # تطبيع القيم
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)
    class_id = np.argmax(predictions)
    confidence = float(np.max(predictions)) * 100

    return jsonify({
        'predicted_class': class_names[class_id],
        'confidence': round(confidence, 2)
    })

if __name__ == '__main__':
    app.run()