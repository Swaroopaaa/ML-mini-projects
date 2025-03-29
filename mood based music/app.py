from flask import Flask, render_template, jsonify
import cv2
import os
import random
import pygame
from deepface import DeepFace

app = Flask(__name__, template_folder="templates")  # Ensure Flask looks in 'templates/'

# Initialize Pygame for playing music
pygame.mixer.init()

# Mood-based song selection
SONGS_FOLDER = "songs"
MOOD_SONGS = {
    "happy": ["happy1.mp3", "happy2.mp3"],
    "sad": ["sad1.mp3", "sad2.mp3"],
    "angry": ["angry1.mp3", "angry2.mp3"],
    "neutral": ["neutral1.mp3", "neutral2.mp3"]
}

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/detect_mood", methods=["POST"])
def detect_mood():
    try:
        # Capture image from webcam
        cam = cv2.VideoCapture(0)
        ret, frame = cam.read()
        cam.release()

        if not ret:
            return jsonify({"error": "Could not capture image. Please check your camera."})

        image_path = "face.jpg"
        cv2.imwrite(image_path, frame)

        # Ensure the image is saved properly
        if not os.path.exists(image_path):
            return jsonify({"error": "Image not saved properly!"})

        # Detect mood using DeepFace
        try:
            result = DeepFace.analyze(image_path, actions=['emotion'], enforce_detection=False)
        except Exception as e:
            return jsonify({"error": f"DeepFace error: {str(e)}"})

        mood = result[0]['dominant_emotion']

        # Map detected mood to predefined categories
        mood_category = "neutral"  # Default
        if mood in ["happy", "joy"]:
            mood_category = "happy"
        elif mood in ["sad", "disgust"]:
            mood_category = "sad"
        elif mood in ["angry", "fear"]:
            mood_category = "angry"

        # Play a song based on detected mood
        if mood_category in MOOD_SONGS:
            song_path = os.path.join(SONGS_FOLDER, random.choice(MOOD_SONGS[mood_category]))
            pygame.mixer.music.load(song_path)
            pygame.mixer.music.play()
            return jsonify({"mood": mood, "song": os.path.basename(song_path)})
        else:
            return jsonify({"error": "No songs available for this mood."})

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "_main_":
    app.run(debug=True)