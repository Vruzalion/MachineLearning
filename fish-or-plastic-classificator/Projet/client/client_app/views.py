from re import I

from importlib_metadata import files
from flask import Flask, render_template, request
import tensorflow as tf
import pickle
import numpy as np

app = Flask(__name__)

i = 0

@app.route('/', methods=['GET'])
@app.route('/classify',  methods=['POST'])
def index():
    if(request.method == "POST"):
        inputChoice = request.form['img_source']
        
        if(inputChoice == "1"):
            #input from a link
            image_url = request.form['img_url']

            result = classify_from_image_url("blob:http://localhost:8001/89b1f3b5-9312-401a-8324-dd01514fb75c")

        else:
            print(dir(request.form['test_file']))
        
        return render_template('index.html', result=result)

    return render_template('index.html')

#function to classify using the image's url
def classify_from_image_url(image_url):
    #chargement du modèle
    #file1 = open("client_app/static/models/classifier.sav", 'rb')
    #model = pickle.load(file1)

    model = tf.keras.models.load_model("client_app/static/saved_model/my_model")
    
    global i

    image_path = tf.keras.utils.get_file("image" + str(i) , origin = image_url)

    i+=1

    img_height, img_width = 200, 200
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