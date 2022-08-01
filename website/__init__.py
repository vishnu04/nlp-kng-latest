from flask import Flask
from os import path
from flask_session import Session

__all__ = [ 
    'actions',
    'nlp_kng',
    'static',
    'templates',
    'answers',
    'models'
]

def create_app():
    app = Flask(__name__)
    app.config['SECRET_KEY'] = 'asdfasdf asdfsadsg'
    app.config['SESSION_PERMANENET'] = False
    app.config['SESSION_TYPE'] = "filesystem"
    # app.config['IMAGE_FOLDER'] = '/static/images'
    Session(app)
    from .actions import actions
    app.register_blueprint(actions, url_prefix='/')
    return app
