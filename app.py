from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.vgg16 import preprocess_input
import numpy as np
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploaded'

# Load the model
model = load_model("vgg16_model1.h5")

# Replace with actual 75-class dictionary
labels = {
    0: 'ADONIS', 1: 'AFRICAN GIANT SWALLOWTAIL', 2: 'AMERICAN SNOOT', 3: 'AN 88', 4: 'APPOLLO',
    5: 'ATALA', 6: 'BANDED ORANGE HELICONIAN', 7: 'BANDED PEACOCK', 8: 'BECKERS WHITE', 9: 'BLACK HAIRSTREAK',
    10: 'BLUE MORPHO', 11: 'BLUE SPOTTED CROW', 12: 'BROWN SIPROETA', 13: 'CABBAGE WHITE', 14: 'CAIRNS BIRDWING',
    15: 'CHECQUERED SKIPPER', 16: 'CHESTNUT', 17: 'CLEOPATRA', 18: 'CLODIUS PARNASSIAN', 19: 'CLOUDED SULPHUR',
    20: 'COMMON BANDED AWL', 21: 'COMMON WOOD-NYMPH', 22: 'COPPER TAIL', 23: 'CRECENT', 24: 'CRIMSON PATCH',
    25: 'DANAID EGGFLY', 26: 'EASTERN COMA', 27: 'EASTERN DAPPLE WHITE', 28: 'EASTERN PINE ELFIN',
    29: 'ELBOWED PIERROT', 30: 'GOLD BANDED', 31: 'GREAT EGGFLY', 32: 'GREAT JAY', 33: 'GREEN CELLED CATTLEHEART',
    34: 'GREY HAIRSTREAK', 35: 'INDRA SWALLOW', 36: 'IPHICLUS SISTER', 37: 'JULIA', 38: 'LARGE MARBLE',
    39: 'MALACHITE', 40: 'MANGROVE SKIPPER', 41: 'MESTRA', 42: 'METALMARK', 43: 'MILBERTS TORTOISESHELL',
    44: 'MONARCH', 45: 'MOURNING CLOAK', 46: 'ORANGE OAKLEAF', 47: 'ORANGE TIP', 48: 'ORCHARD SWALLOW',
    49: 'PAINTED LADY', 50: 'PAPER KITE', 51: 'PEACOCK', 52: 'PINE WHITE', 53: 'PIPEVINE SWALLOW',
    54: 'POPINJAY', 55: 'PURPLE HAIRSTREAK', 56: 'PURPLISH COPPER', 57: 'QUESTION MARK', 58: 'RED ADMIRAL',
    59: 'RED CRACKER', 60: 'RED POSTMAN', 61: 'RED SPOTTED PURPLE', 62: 'SCARCE SWALLOW', 63: 'SILVER SPOT SKIPPER',
    64: 'SLEEPY ORANGE', 65: 'SOOTYWING', 66: 'SOUTHERN DOGFACE', 67: 'STRAITED QUEEN', 68: 'TROPICAL LEAFWING',
    69: 'TWO BARRED FLASHER', 70: 'ULYSES', 71: 'VICEROY', 72: 'WOOD SATYR', 73: 'YELLOW SWALLOW TAIL',
    74: 'ZEBRA LONG WING'
}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def upload_predict():
    if request.method == 'POST':
        image_file = request.files['image']
        if image_file:
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], image_file.filename)
            image_file.save(image_path)

            img = load_img(image_path, target_size=(224, 224))
            x = img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x = preprocess_input(x)

            preds = model.predict(x)
            class_index = np.argmax(preds)
            class_label = labels.get(class_index, "Unknown")

            return render_template('result.html', image_path=image_path, label=class_label)

    return render_template('predict.html')

if __name__ == '__main__':
    app.run(debug=True)
