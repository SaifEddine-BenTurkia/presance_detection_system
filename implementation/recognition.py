
from deepface import DeepFace
import numpy as np
import cv2

class FaceNet512:
    def __init__(self):
        # DeepFace will automatically download and load the model on first use
        self.model_name = "Facenet512"
        self.input_size = (160, 160)
        
    def preprocess(self, face_img):
        # DeepFace handles preprocessing internally, but if you need custom preprocessing:
        resized = cv2.resize(face_img, self.input_size)
        normalized = (resized - 127.5) / 128.0
        return normalized
    
    def get_embedding(self, face_img):
        # Option 1: Let DeepFace handle everything (recommended)
        # DeepFace.represent expects the full image and will detect faces automatically
        # If you already have a cropped face, you might need to handle it differently
        try:
            embedding = DeepFace.represent(
                img_path=face_img,  # Can be image array or path
                model_name=self.model_name,
                enforce_detection=False  # Set to False if you're sure it's a face
            )[0]["embedding"]
            
            return np.array(embedding)
        except:
            # Fallback: if face_img is already a numpy array (cropped face)
            return self._get_embedding_from_array(face_img)