import cv2
import random
import os
import pygame
from deepface import DeepFace

# Initialize Pygame for playing music
pygame.mixer.init()

# Capture image from webcam
cam = cv2.VideoCapture(0)
ret, frame = cam.read()
image_path = "face.jpg"
cv2.imwrite(image_path, frame)
cam.release()

# Detect mood
result = DeepFace.analyze(image_path, actions=['emotion'], enforce_detection=False)
mood = result[0]['dominant_emotion']
print("Detected Mood:", mood)

# Folder containing songs
songs_folder = "songs"

# Mood-based song selection
mood_songs = {
    "happy": ["happy1.mp3", "happy2.mp3"],
    "sad": ["sad1.mp3", "sad2.mp3"],
    "angry": ["angry1.mp3", "angry2.mp3"],
    "neutral": ["neutral1.mp3", "neutral2.mp3"]
}

# Match detected mood with available moods
if mood in ["happy", "joy"]:
    mood_category = "happy"
elif mood in ["sad", "disgust"]:
    mood_category = "sad"
elif mood in ["angry", "fear"]:
    mood_category = "angry"
else:
    mood_category = "neutral"

# Play a song based on detected mood
if mood_category in mood_songs:
    song_path = os.path.join(songs_folder, random.choice(mood_songs[mood_category]))
    print("Playing song:", song_path)
    pygame.mixer.music.load(song_path)
    pygame.mixer.music.play()
else:
    print("No songs available for this mood.")

# Keep playing until user stops
input("Press Enter to stop the music...")
pygame.mixer.music.stop()

