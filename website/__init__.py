from flask import Flask
from os import path, removedirs
from flask_session import Session
from flask_socketio import SocketIO, emit
from .nlp_kng.create_temp_dir import directory_name

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
    socketio = SocketIO(app)
    @socketio.on('disconnect')
    def remove_files():
        emit(
            'user disconnected', 
            broadcast=False
        )
        tmpdir = directory_name
        print(f'remove_files: {tmpdir}')
        try:
            removedirs(tmpdir)
        except:
            print('Error in deleting the file:{tmpdir}')
    # app.config['IMAGE_FOLDER'] = '/static/images'
    Session(app)
    from .actions import actions
    app.register_blueprint(actions, url_prefix='/')
    return app

    


    