from website import create_app
import os
from website.nlp_kng import config

os.environ["CORENLP_HOME"] = config.CORENLP_DIR
app = create_app()


if __name__ == '__main__':
    print(f'calling main ')
    app.run(debug=True, host='0.0.0.0')
    # app.run(debug = True)

    