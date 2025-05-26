import cv2
from deepface import DeepFace
import numpy as np
import pandas as pd
import time
from scipy.spatial.distance import cosine
import torch 


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


if torch.cuda.is_available():
    torch.cuda.empty_cache()

# Load embeddings with normalization
embedding_df = pd.read_csv("face_embeddings.csv")
known_embeddings = []
for embedding_str in embedding_df['embedding']:
    arr = np.fromstring(embedding_str, sep=' ')
    known_embeddings.append(arr)
known_embeddings = [emb / np.linalg.norm(emb) for emb in known_embeddings]  # Normalize

if not known_embeddings:
    raise ValueError("No valid embeddings in database")



cap = cv2.VideoCapture(0)
threshold = 0.7 # Adjusted threshold


desired_fps = 10
frame_time = 1000 // desired_fps


#build the model to lighten the load on the GPU


while True:
    ret, frame = cap.read()
    if not ret:
        break

    try:
        faces = DeepFace.extract_faces(frame, detector_backend = "retinaface" , align = True)
    except:
        faces = []

    if faces:
        for face in faces:
            face_img = face["face"]
            
            # Generate and normalize live embedding
            embedding = np.array(DeepFace.represent(
                face_img, 
                model_name = "ArcFace",  
                enforce_detection=False
            )[0]["embedding"])
            embedding = embedding / np.linalg.norm(embedding)

            # Calculate similarities
            similarities = [1 - np.dot(embedding - stored_emb) for stored_emb in known_embeddings]
            
            if similarities:  # Extra safety check
                  # Compute the median similarity
                best_match_idx = np.argmax(similarities)
                best_similarity = similarities[best_match_idx]
                
                label = ("Unauthorized" if best_similarity > threshold 
                         else f"Recognized: {embedding_df.iloc[best_match_idx]['name']} ({best_similarity:.2f})")
                
                # Draw bounding box
                area = face["facial_area"]
                x, y, w, h = area["x"], area["y"], area["w"], area["h"]
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 1)
                cv2.putText(frame, label, (x, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 1)
    else:
        cv2.putText(frame, "No face detected", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 1)

    cv2.imshow("Face Recognition", frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()