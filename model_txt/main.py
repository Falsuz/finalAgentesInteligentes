import pandas as pd
import nltk
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf

# Descargar el tokenizador de NLTK
nltk.download('punkt')

# Cargar los datasets de entrenamiento y validación
training_data_path = "model_txt\\twitter_training.csv"
validation_data_path = "model_txt\\twitter_validation.csv"

# Leer los datasets y asignar nombres a las columnas
column_names = ["id", "category", "label", "text"]
training_data = pd.read_csv(training_data_path, header=None, names=column_names)
validation_data = pd.read_csv(validation_data_path, header=None, names=column_names)

# Concatenar los datasets para entrenamiento
data = pd.concat([training_data, validation_data])

# Extraer texto y etiquetas
tweets = data['text'].tolist()
labels = data['label'].tolist()

# Tokenizar los tweets
x = [word_tokenize(tweet.lower()) for tweet in tweets]

# Asignar etiquetas numéricas
# Asumiendo que "Positive" es positivo, "Negative" es negativo, y se puede extender si hay "Neutral"
label_mapping = {"Positive": 2, "Neutral": 1, "Negative": 0}
y = np.array([label_mapping[label] for label in labels])

# Entrenar el modelo Word2Vec
model_w2v = Word2Vec(
    sentences=x,
    vector_size=10,
    window=3,
    min_count=1,
    workers=4
)

# Función para convertir oraciones en vectores
def sentence_to_vector(sentence):
    return np.mean(
        [model_w2v.wv[word] for word in sentence if word in model_w2v.wv],
        axis=0
    )

x_vectors = np.array([sentence_to_vector(sentence) for sentence in x])

# Dividir datos en entrenamiento y prueba
x_train, x_test, y_train, y_test = train_test_split(
    x_vectors,
    y,
    test_size=0.2,
    random_state=42
)

# Crear el modelo de clasificación con TensorFlow
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(32, activation='relu', input_shape=(x_train.shape[1],)),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(3, activation='softmax')  # 3 clases: negativo, neutral, positivo
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Entrenar el modelo
history = model.fit(
    x_train, y_train,
    epochs=50,
    validation_data=(x_test, y_test),
    batch_size=16
)

# Evaluar el modelo
loss, acc = model.evaluate(x_test, y_test)
print(f"Loss: {loss}")
print(f"Accuracy: {acc}")

# Predicción de una nueva oración
new_sentence = "I'm getting on Borderlands, and I will destroy everyone!"
new_sentence_tokenized = word_tokenize(new_sentence.lower())
new_sentence_vector = sentence_to_vector(new_sentence_tokenized)
new_sentence_vector = np.array([new_sentence_vector])

prediction = model.predict(new_sentence_vector)
predicted_label = np.argmax(prediction)
reverse_label_mapping = {v: k for k, v in label_mapping.items()}

print(f"Prediction probabilities: {prediction}")
print(f"Predicted class: {reverse_label_mapping[predicted_label]}")
