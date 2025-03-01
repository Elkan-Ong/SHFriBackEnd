import tensorflow as tf
import numpy as np
from PIL import Image
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
import os
import keras
import json
import pickle
from .ml_models import FakeNewsModel  # Import from the new module


import cv2
import numpy as np
IMG_SIZE = 224

def convert_to_frequency_domain(image):
    image_uint8 = (image * 255).astype(np.uint8)
    gray = cv2.cvtColor(image_uint8, cv2.COLOR_RGB2GRAY)
    dft = np.fft.fft2(gray)
    dft_shift = np.fft.fftshift(dft)
    magnitude_spectrum = 20 * np.log(np.abs(dft_shift) + 1e-8)
    magnitude_spectrum = cv2.resize(magnitude_spectrum, (IMG_SIZE, IMG_SIZE))
    magnitude_spectrum = magnitude_spectrum / np.max(magnitude_spectrum)
    magnitude_spectrum = np.expand_dims(magnitude_spectrum, axis=-1)
    return magnitude_spectrum

def preprocess_test_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Error: Unable to load image from {image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    image = image / 255.0
    freq_map = convert_to_frequency_domain(image)
    image = np.expand_dims(image, axis=0)
    freq_map = np.expand_dims(freq_map, axis=0)
    return [image, freq_map]

model_path = "./model/fake_image_detection_model.keras"
model = keras.models.load_model(model_path)


text_model_path = os.path.join(os.path.dirname(__file__), "fake_news_model.pkl")
print(text_model_path)
with open(text_model_path, "rb") as f:
    text_model = pickle.load(f)

@csrf_exempt
def upload_image(request):
    if request.method == "POST" and request.FILES.get("image"):
        image = request.FILES["image"]

        # Save the uploaded image temporarily
        file_path = default_storage.save(f"uploads/{image.name}", ContentFile(image.read()))
        full_path = default_storage.path(file_path)  # Get absolute path
        print(full_path)
        try:
            # Load the CNN model for prediction

            # Preprocess the image before prediction
            image = preprocess_test_image(full_path)

            # Make a prediction
            prediction = model.predict(image)
            print(prediction)
            prediction = "Real" if prediction[0][0] < 0.5 else "Fake"  # Example output decoding
            # Delete the file after processing
            if os.path.exists(full_path):
                os.remove(full_path)

            return JsonResponse({"message": "Image processed", "prediction": prediction})

        except Exception as e:
            print(e)
            return JsonResponse({"error": str(e)}, status=500)

    return JsonResponse({"error": "No image provided"}, status=400)


@csrf_exempt
def predict_text(request):
    if request.method == "POST":
        try:
            # Parse the request body to extract the input text
            data = json.loads(request.body)
            input_text = data.get("text", None)

            if not input_text:
                return JsonResponse({"error": "No text input provided"}, status=400)



            # Example: prediction = your_nlp_model.predict(input_text)
            prediction = text_model.predict_fake_news(input_text)
            print(prediction)
            return JsonResponse({"input": input_text, "prediction": prediction})

        except json.JSONDecodeError:
            return JsonResponse({"error": "Invalid JSON input"}, status=400)
        except Exception as e:
            print(e)
            return JsonResponse({"error": str(e)}, status=500)

    return JsonResponse({"error": "Invalid request method. Use POST."}, status=405)