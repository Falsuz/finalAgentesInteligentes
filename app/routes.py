from flask import Flask, render_template, request, redirect, url_for, jsonify
import os
from PIL import Image
from io import BytesIO
import base64
from werkzeug.utils import secure_filename
from flask_sqlalchemy import SQLAlchemy

app = Flask(__name__)

# Configuración de la base de datos
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Configuración de la carpeta para subir archivos
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Configuración de las extensiones permitidas
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Crear instancia de SQLAlchemy
db = SQLAlchemy(app)

class Post(db.Model):
    __tablename__ = 'posts'
    id = db.Column(db.Integer, primary_key=True)
    image_path = db.Column(db.String(255), nullable=False)  # Ruta de la imagen guardada
    description = db.Column(db.Text, nullable=False)
    image_sentiment = db.Column(db.String(50), nullable=True)  # Predicción del modelo CNN
    text_sentiment = db.Column(db.String(50), nullable=True)  # Predicción del modelo NLP

    def __repr__(self):
        return f"<Post with id: {self.id}>"

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            print('No file attached in request')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            print('No file selected')
            return redirect(request.url)
        if file and check_allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            # Proceso para realizar predicción de la imagen (placeholder por ahora)
            image_sentiment = "AQUI va la predicción"
            text_sentiment = "AQUI va la predicción"
            post = Post(
                image_path=file_path,
                description=request.form.get('description', ''),
                image_sentiment=image_sentiment,
                text_sentiment=text_sentiment
            )
            db.session.add(post)
            db.session.commit()
            # Convertimos la imagen en base64 para mostrarla en la plantilla
            img_data = convert_image_to_base64(file_path)
            return render_template('upload.html', img_data=img_data, post=post)
    return render_template('index.html', img_data="")

# Función para verificar si la extensión del archivo es permitida
def check_allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Función para convertir imagen a base64
def convert_image_to_base64(image_path):
    with open(image_path, "rb") as img_file:
        encoded_string = base64.b64encode(img_file.read()).decode()
    return encoded_string

if __name__ == "__main__":
    with app.app_context():
        db.create_all()  # Crear las tablas si no existen
    app.run(debug=True)
