import numpy as np
from scipy.io.wavfile import write
import uuid

def generate_music(mood, genre):

    rate = 44100
    duration = 5

    if mood == "happy":
        freq = 600
    elif mood == "sad":
        freq = 300
    elif mood == "energetic":
        freq = 900
    else:
        freq = 500

    if genre == "lofi":
        freq -= 100
    elif genre == "edm":
        freq += 200

    t = np.linspace(0, duration, int(rate * duration), False)

    audio = np.sin(2 * np.pi * freq * t)

    filename = f"music_{uuid.uuid4().hex}.wav"

    write(filename, rate, audio.astype(np.float32))

    return filename
