import tensorflow as tf
import numpy as np
from PIL import Image
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# Carga de modelos REEMPLAZAR POR MODELOS REALES
"""text_model = tf.keras.models.load_model('fashion_mnist.h5')
image_model = tf.keras.models.load_model('fashion_mnist.h5')
"""

# def predict_image_sentiment(image_path):
#     """
#     Realiza una predicción de sentimiento en una imagen y las pre procesa
#     """
#     img = Image.open(image_path).resize((224, 224))  # Ajusta según tu modelo
#     img_array = np.array(img) / 255.0
#     img_array = np.expand_dims(img_array, axis=0)  # Agregar dimensión para batch
#     prediction = image_model.predict(img_array)
#     labels = ["negative", "neutral", "positive"]
#     return labels[np.argmax(prediction)]

# def predict_text_sentiment(description):
#     """
#     Realiza una predicción de sentimiento en un texto.
#     """
#     processed_text = preprocess_text(description)
#     prediction = text_model.predict([processed_text])  # Ajustar según el input del modelo
#     labels = ["negative", "neutral", "positive"]
#     return labels[np.argmax(prediction)]

# Preprocesamiento
def preprocess_text(text):
    """
    Limpia y tokeniza el texto para análisis de sentimientos.
    """
    # Convertir a minúsculas
    text = text.lower()

    # Eliminar URLs
    text = re.sub(r"http\S+|www\S+|https\S+", '', text, flags=re.MULTILINE)

    # Eliminar menciones y hashtags (TAMPOCO ME DA BUENA VIBRA )
    # text = re.sub(r"@\w+|#\w+", '', text)

    # Tokenización
    tokenized_text = word_tokenize(text)

    # Después de tokenizar elimina caracteres como: #
    tokenized_text = [word.replace('#', '') for word in tokenized_text]

    # Eliminar palabras vacías
    stop_words = set(stopwords.words('spanish'))  # Cambiar a 'spanish' si es necesario
    filtered_text = [word for word in tokenized_text if word not in stop_words]

    return filtered_text

tweet = "El servicio de #atención al cliente fue excelente. ¡Gracias @empresa! 😊 #Recomendado"
resultado = preprocess_text(tweet)
print(resultado)
