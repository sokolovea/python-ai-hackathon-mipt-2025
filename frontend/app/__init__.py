from flask import Flask
from .routes import register_routes
import os

def create_app():
    app = Flask(__name__)
    app.secret_key = os.getenv("FLASK_SECRET_KEY", "supersecretkey")
    register_routes(app)
    return app