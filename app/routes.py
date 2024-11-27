
from flask import render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
import os
import base64
from models import db, Post
from config import Config

# Función para verificar si la extensión del archivo es permitida
def check_allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in Config.ALLOWED_EXTENSIONS

# Función para convertir imagen a base64
def convert_image_to_base64(image_path):
    with open(image_path, "rb") as img_file:
        encoded_string = base64.b64encode(img_file.read()).decode()
    return encoded_string

# Rutas y vistas
def init_app(app):
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
                file_path = os.path.join(Config.UPLOAD_FOLDER, filename)
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
