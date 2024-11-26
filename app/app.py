# app.py

from flask import Flask
from models import db
from config import Config
from routes import init_app

app = Flask(__name__)

# Configurar la aplicación con la clase Config
app.config.from_object(Config)

# Inicializar la base de datos
db.init_app(app)

# Inicializar las rutas desde vistas.py
init_app(app)

# Crear las tablas en la base de datos si no existen
with app.app_context():
    db.create_all()

if __name__ == "__main__":
    app.run(debug=True)
