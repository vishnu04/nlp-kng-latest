from website import create_app
import os

app = create_app()

if __name__ == '__main__':
    print(f'calling main')
    app.run(debug=True, host='0.0.0.0')
    

    