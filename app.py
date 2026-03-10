from flask import Flask, render_template, request, send_file
from music_ai import generate_music

app = Flask(__name__)

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/generate', methods=['POST'])
def generate():

    mood = request.form['mood']
    genre = request.form['genre']

    file_path = generate_music(mood, genre)

    return send_file(file_path, as_attachment=True)

if __name__ == "__main__":
    app.run()

