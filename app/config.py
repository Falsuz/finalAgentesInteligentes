import os

class Config:
    # Configuración de la base de datos
    SQLALCHEMY_DATABASE_URI = 'sqlite:///database.db'
    SQLALCHEMY_TRACK_MODIFICATIONS = False

    # Configuración de la carpeta para subir archivos
    UPLOAD_FOLDER = 'uploads'
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)

    # Configuración de las extensiones permitidas
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
