from website import create_app


app = create_app()

if __name__ == '__main__':
    print('calling main')
    app.run(debug=True)