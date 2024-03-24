from flask import Flask, jsonify, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from flask_cors import CORS
import cv2
import imutils

app = Flask(__name__)
CORS(app)

brain_tumor_model = load_model("../final_model.h5")

@app.route('/api/v1/models/brain_tumor', methods=["POST"])
def detect_brain_tumor():
    data = request.files['image']
    image = process_image(data)
    prediction = brain_tumor_model.predict(image)
    result = numToLabel(np.argmax(prediction))
    return jsonify(result), 200

def process_image(file_storage, target_size=(200, 200)):
    loaded_img = Image.open(file_storage)

    cv2_img = np.array(loaded_img).astype(np.uint8)
    cv2_img = crop_img(cv2_img)
    cv2_img = apply_filters(cv2_img)
    image_array = cv2_img.astype(np.float64) / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    return image_array

def crop_img(img):
	gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
	gray = cv2.GaussianBlur(gray, (3, 3), 0)

	# threshold the image, then perform a series of erosions +
	# dilations to remove any small regions of noise
	thresh = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY)[1]
	thresh = cv2.erode(thresh, None, iterations=2)
	thresh = cv2.dilate(thresh, None, iterations=2)

	# find contours in thresholded image, then grab the largest one
	cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	cnts = imutils.grab_contours(cnts)
	c = max(cnts, key=cv2.contourArea)

	# find the extreme points
	extLeft = tuple(c[c[:, :, 0].argmin()][0])
	extRight = tuple(c[c[:, :, 0].argmax()][0])
	extTop = tuple(c[c[:, :, 1].argmin()][0])
	extBot = tuple(c[c[:, :, 1].argmax()][0])
	ADD_PIXELS = 0
	new_img = img[extTop[1]-ADD_PIXELS:extBot[1]+ADD_PIXELS, extLeft[0]-ADD_PIXELS:extRight[0]+ADD_PIXELS].copy()
	
	return new_img

def apply_filters(img, image_size=(200, 200)):
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    img = cv2.bilateralFilter(img, 2, 50, 50)
    img = cv2.applyColorMap(img, cv2.COLORMAP_BONE)
    img = cv2.resize(img, image_size)
    return img

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
    
