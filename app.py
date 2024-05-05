from flask import Flask, render_template, request, jsonify
from PIL import Image
import numpy as np
import io
from tensorflow import keras
import tensorflow as tf
import base64


app = Flask(__name__)

# Load the model
loaded_model = keras.models.load_model('zero_dce_model')

def allowed_file(filename):
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS



def infer(original_image):
    image = keras.preprocessing.image.img_to_array(original_image)
    image = image[:, :, :3] if image.shape[-1] > 3 else image
    image = image.astype("float32") / 255.0
    image = np.expand_dims(image, axis=0)
    output_image = loaded_model(image)
    try:
        # Assuming output_image is a NumPy array
        output_image = tf.cast((output_image[0, :, :, :] * 255), dtype=np.uint8)
        output_image = tf.keras.preprocessing.image.array_to_img(output_image)
        return output_image
    except Exception as e:
        print("Error converting output image:", e)
        return None


# Function to enhance a single image
def enhance_single_image(image):
    enhanced_image = infer(image)
    return enhanced_image

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/enhance', methods=['POST'])
def enhance():
    # Check if request contains file
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']

    # Check if file is empty
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    # Check if file is an image
    if file and allowed_file(file.filename):
        # Read image
        img = Image.open(io.BytesIO(file.read()))

        # Enhance image
        enhanced_image = enhance_single_image(img)

        # Convert enhanced image to bytes
        img_byte_array = io.BytesIO()
        enhanced_image.save(img_byte_array, format='PNG')
        img_byte_array = img_byte_array.getvalue()

        # Convert bytes to base64 string
        enhanced_base64_string = base64.b64encode(img_byte_array).decode('utf-8')

        return jsonify({'enhanced_image': enhanced_base64_string})

    else:
        return jsonify({'error': 'Invalid file format'})

if __name__ == '__main__':
    app.run(debug=True)
