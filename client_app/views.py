from re import I

from importlib_metadata import files
from flask import Flask, render_template, request
import tensorflow as tf
import pickle
import numpy as np
import time

app = Flask(__name__)


model = tf.keras.models.load_model("client_app/static/saved_model/my_model")

@app.route('/', methods=['GET'])
@app.route('/classify',  methods=['POST'])
def index():
    if(request.method == "POST"):
        print("________________post")
        inputChoice = request.form['img_source']
        print(inputChoice)

        if(inputChoice == "1"):
            #input from a link
            image_url = request.form['img_url']
           
            if(image_url):
                result = classify_from_image_url(image_url)

            else:
                result = "Please select an image to classify"

            return render_template('index.html', result=result, image_url = image_url)

        else:
            image_file = request.form['img_file']

            if(image_file):

                try:
                    result = classify_from_local_image(image_file)

                except:
                    result = "Enter a valid image name"

            else:
                result = "Please select an image to classify"
        
            return render_template('index.html', result=result, image_url="static/img/" + image_file)

    return render_template('index.html')

#function to classify using the image's url
def classify_from_image_url(image_url):
    global model

    milliseconds = int(round(time.time() * 1000))

    image_path = tf.keras.utils.get_file(str(milliseconds), origin = image_url)

    img_height, img_width = 80, 80
    class_names = ["fish", "plastic"]
    #"client_app/static/img/img.png"
    img = tf.keras.utils.load_img(
        image_path, target_size=(img_height, img_width)
    )
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)
    
    predictions = model.predict(img_array)

    score = tf.nn.softmax(predictions[0])

    return ( "This image most likely belongs to {} with a {:.2f} percent confidence."
    .format(class_names[np.argmax(score)], 100 * np.max(score)))


#function to classify a local image
def classify_from_local_image(image_file):
    global model

    milliseconds = int(round(time.time() * 1000))

    image_path = "client_app/static/img/" + image_file

    img_height, img_width = 80, 80
    class_names = ["fish", "plastic"]
    
    img = tf.keras.utils.load_img(
        image_path, target_size=(img_height, img_width)
    )
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)
    
    predictions = model.predict(img_array)

    score = tf.nn.softmax(predictions[0])

    return ( "This image most likely belongs to {} with a {:.2f} percent confidence."
    .format(class_names[np.argmax(score)], 100 * np.max(score)))
    

if __name__ == "__main__":
    app.run(port=8001, debug=True)