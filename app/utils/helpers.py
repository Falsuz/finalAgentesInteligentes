from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import re

# Carga de modelos REEMPLAZAR POR MODELOS REALES
image_model = load_model('path_to_cnn_model.h5')
text_model = load_model('path_to_nlp_model.h5')

def predict_image_sentiment(image_path):
    """
    Realiza una predicción de sentimiento en una imagen y las pre procesa
    """
    img = Image.open(image_path).resize((224, 224))  # Ajusta según tu modelo
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # Agregar dimensión para batch
    prediction = image_model.predict(img_array)
    labels = ["negative", "neutral", "positive"]
    return labels[np.argmax(prediction)]

def predict_text_sentiment(description):
    """
    Realiza una predicción de sentimiento en un texto.
    """
    processed_text = preprocess_text(description)
    prediction = text_model.predict([processed_text])  # Ajustar según el input del modelo
    labels = ["negative", "neutral", "positive"]
    return labels[np.argmax(prediction)]

def preprocess_text(text):
    """
    Preprocesa el texto para el modelo NLP.
    USAR NLTK PARA TOKENIZAR, VECTORES, ETC, ETC...
    """
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)  # Eliminar puntuación
    text = text.strip()
    return text
