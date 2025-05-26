# core/detection.py
import cv2
import time
import os
import pygame  # For sound alerts
import tempfile
import numpy as np
from datetime import datetime
from database import ESClient
from presence_logs import PresenceTracker
from deepface import DeepFace
from config import *
from retinaface import RetinaFace

def detect_faces(frame):
    """Detect faces using RetinaFace and return coordinates"""
    faces = []
    obj = RetinaFace.detect_faces(frame)
    if isinstance(obj, dict):
        for key in obj.keys():
            identity = obj[key]
            facial_area = identity["facial_area"]
            y1, x1, y2, x2 = facial_area
            w = x2 - x1
            h = y2 - y1
            faces.append((x1, y1, w, h))
    return faces

class FaceNet512:
    def __init__(self):
        self.model_name = "Facenet512"

    def get_embedding(self, face_img):
        try:
            result = DeepFace.represent(
                img_path = face_img,  # pass numpy array directly
                model_name=self.model_name,
                enforce_detection=False
            )
            embedding = np.array(result[0]["embedding"])
            return embedding
        except Exception as e:
            print(f"Error getting embedding: {e}")
            return None

class FaceMonitor:
    def __init__(self):
        self.cap = cv2.VideoCapture(VIDEO_SOURCE)
        self.out = None
        self.tracker = PresenceTracker()
        self.db = ESClient()
        self.face_model = RetinaFace
        if PLAY_ALERT_SOUND:
            pygame.mixer.init()
            self.alert_sound = pygame.mixer.Sound(ALERT_SOUND_PATH)
        else:
            self.alert_sound = None
        self.frame_width = int(self.cap.get(3))
        self.frame_height = int(self.cap.get(4))
        
        # Ensure output folder exists
        os.makedirs("implementation/output", exist_ok=True)
        os.makedirs("logs", exist_ok=True)

        self.video_writer = cv2.VideoWriter(
            "implementation/output/output.mp4",
            cv2.VideoWriter_fourcc(*'mp4v'),
            20,
            (self.frame_width, self.frame_height)
        )

    def process_stream(self):
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                print("End of video or failed to grab frame.")
                break
            
            faces = RetinaFace.detect_faces(frame)
            if isinstance(faces, dict):  # If faces are detected
                for key, face in faces.items():
                    facial_area = face["facial_area"]
                    x1, y1, x2, y2 = facial_area
                    face_img = frame[y1:y2, x1:x2]
                    embedding = self.extract_embedding(face_img)

                    label = "Unknown"
                    if embedding is not None:
                        result = self.db.search_face(embedding.tolist())
                        if result:
                            label = result["_source"]["person_name"]
                            self.tracker.update_presence(label)
                        else:
                            label = "Unauthorized"

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, label, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

                    if label == "Unauthorized" and self.alert_sound:
                        self.alert_sound.play()

            self.video_writer.write(frame)

            if SHOW_LIVE_FEED:
                cv2.imshow("Live Feed", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        print("[INFO] Finished processing stream.")

    def extract_embedding(self, face_img):
        # Dummy extractor for now
        try:
            return np.random.rand(512)
        except:
            return None

    def cleanup(self):
        print("[INFO] Cleaning up...")
        self.cap.release()
        self.video_writer.release()
        cv2.destroyAllWindows()
        self.tracker.save_logs()
