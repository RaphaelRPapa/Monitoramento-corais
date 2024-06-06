from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import numpy as np
import io

app = Flask(__name__)

# Carregar o modelo treinado
model = load_model('coral_model.h5')

def preprocess_image(img):
    img = img.resize((128, 128))  # ajuste conforme necessário
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = img / 255.0  # normalização
    return img

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['file']
    img = Image.open(io.BytesIO(file.read()))
    processed_img = preprocess_image(img)
    
    prediction = model.predict(processed_img)
    print(f"Prediction raw output: {prediction}")  # Diagnóstico: ver a saída crua
    
    if prediction[0] > 0.5:
        result = "O coral está saudável"
    else:
        result = "O coral não está saudável"
    
    return jsonify({'result': result})

if __name__ == '__main__':
    app.run(debug=True)
