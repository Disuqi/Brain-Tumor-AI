from flask import Flask, jsonify, request
from tensorflow.keras.models import load_model

import numpy as np
from flask_cors import CORS

from image_processing import process_image

app = Flask(__name__)
CORS(app)

brain_tumor_model = load_model("../final_model.h5")

@app.post('/api/v1/models/brain_tumor')
def detect_brain_tumor():
    data = request.files['image']
    image = process_image(data)
    prediction = brain_tumor_model.predict(image)
    result = numToLabel(np.argmax(prediction))
    return jsonify(result), 200


def numToLabel(num):
    if num == 0:
        return "Glioma"
    elif num == 1:
        return "Meningioma"
    elif num == 2:
        return "No Tumor"
    elif num == 3:
        return "Pituitary"

if __name__ == '__main__':
    app.run(debug=True)
    
