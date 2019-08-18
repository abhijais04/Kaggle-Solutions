import os
import numpy as np
from keras.preprocessing import image
from keras.models import load_model
from flask import Flask , request, jsonify

app = Flask(__name__)

UPLOAD_FOLDER = os.path.basename('./')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
img_width, img_height = 28, 28

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

@app.route("/predict", methods=['POST'])
def predict():

	#only works if the image has black background and the digit in white
	file = request.files['file']
	f = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
	file.save(f)

	img = image.load_img(f, target_size=(img_width, img_height))
	img = image.img_to_array(img)
	img = rgb2gray(img)
	img = img.flatten()
	img = np.expand_dims(img, axis=0)
	#print (img.shape)
	img = img/255.0

	model = load_model('./trained_model/digit_recog_93_95.h5')
	pred = model.predict(img)
	print (pred)
	result = np.argmax(pred, axis=1)
	return jsonify({"Predicted Class": int(result[0])})


@app.route("/hello")
def hello():
	return "hello world !"


if __name__== '__main__':
	app.run(debug=True)

