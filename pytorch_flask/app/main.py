from flask import Flask, jsonify, request

from torch_utils import transform_image, get_prediction

app = Flask(__name__)

ALLOWED_EXTENSIONS = {'png','jpg','jpeg'}
def allowed_file(filename):
    #xxx.png
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
    #split the file from right and split at the dot with max one split and check if it an compatible file

@app.route('/predict', methods=['POST'])
def predict():
    #error checking before running to make sure server does not crash
    if request.method == 'POST':
        file = request.files.get('file')
        #check if file is not found error
        if file is None or file.filename == "":
            return jsonify({'error': 'no file'})
        #check for compatible file as input "jpg, png"
        if not allowed_file(file.filename):
            return jsonify({'error': 'format not supported'})
        
        try:
            img_bytes = file.read()
            tensor = transform_image(img_bytes)
            prediction = get_prediction(tensor)
            data = {'prediction': prediction.item(), 'class_name': str(prediction.item())}
            return jsonify(data)
        
        except:
            return jsonify({'error': 'error during prediction'})


    #1) load image 
    #2) transform image -> tensor
    #3) prediction
    #4) outputs json


