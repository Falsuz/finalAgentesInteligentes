from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()

class Post(db.Model):
    __tablename__ = 'posts'
    id = db.Column(db.Integer, primary_key=True)
    image_path = db.Column(db.String(255), nullable=False)  # Ruta de la imagen guardada
    description = db.Column(db.Text, nullable=False)
    image_sentiment = db.Column(db.String(50), nullable=True)  # Predicción del modelo CNN
    text_sentiment = db.Column(db.String(50), nullable=True)  # Predicción del modelo NLP

    def __repr__(self):
        return f"<Post with id: {self.id}>"
