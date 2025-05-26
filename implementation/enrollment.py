# core/enrollment.py
import tkinter as tk
from tkinter import ttk, messagebox
import cv2
from PIL import Image, ImageTk
import numpy as np
import tempfile
import os
import time
from database import ESClient
from deepface import DeepFace
from retinaface import RetinaFace

class FaceNet512:
    def __init__(self):
        # DeepFace will automatically download and load the model on first use
        self.model_name = "Facenet512"
        
    def get_embedding(self, face_img):
        """
        Get FaceNet embedding for a cropped face image
        """
        try:
            # Create a temporary file for the face image
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_file:
                # Write the face image to temporary file
                cv2.imwrite(tmp_file.name, face_img)
                
                try:
                    # Get embedding using DeepFace
                    result = DeepFace.represent(
                        img_path=tmp_file.name,
                        model_name=self.model_name,
                        enforce_detection=False  # Skip face detection since we already have a cropped face
                    )
                    
                    embedding = np.array(result[0]["embedding"])
                    return embedding
                    
                finally:
                    # Clean up temporary file
                    os.unlink(tmp_file.name)
                    
        except Exception as e:
            print(f"Error getting embedding: {e}")
            return None

def detect_faces(frame):
    """Detect faces using RetinaFace and return coordinates"""
    faces = []
    try:
        obj = RetinaFace.detect_faces(frame)
        if isinstance(obj, dict):
            for key in obj.keys():
                identity = obj[key]
                facial_area = identity["facial_area"]
                y1, x1, y2, x2 = facial_area
                w = x2 - x1
                h = y2 - y1
                faces.append((x1, y1, w, h))
    except Exception as e:
        print(f"Face detection error: {e}")
    return faces

class EnrollmentApp:
    def __init__(self):
        self.window = tk.Tk()
        self.window.title("Face Enrollment")
        self.es = ESClient()
        self.setup_ui()
        self.cap = cv2.VideoCapture(0)
        self.update_camera()
        self.facenet = FaceNet512()
        self.current_frame = None
    
    def setup_ui(self):
        # Form fields
        ttk.Label(self.window, text="Full Name:").grid(row=0, column=0, padx=10, pady=5)
        self.name_entry = ttk.Entry(self.window, width=30)
        self.name_entry.grid(row=0, column=1, padx=10, pady=5)
        
        # Camera preview
        self.cam_label = ttk.Label(self.window, text="Camera Loading...")
        self.cam_label.grid(row=1, column=0, columnspan=2, padx=10, pady=10)
        
        # Status label
        self.status_label = ttk.Label(self.window, text="Ready for enrollment")
        self.status_label.grid(row=2, column=0, columnspan=2, padx=10, pady=5)
        
        # Capture button
        self.capture_btn = ttk.Button(self.window, text="Capture Images (5)", 
                                    command=self.capture_images)
        self.capture_btn.grid(row=3, column=0, columnspan=2, padx=10, pady=10)
        
        # Progress bar
        self.progress = ttk.Progressbar(self.window, length=300, mode='determinate')
        self.progress.grid(row=4, column=0, columnspan=2, padx=10, pady=5)
    
    def update_camera(self):
        ret, frame = self.cap.read()
        if ret:
            self.current_frame = frame.copy()
            
            # Detect faces and draw rectangles
            faces = detect_faces(frame)
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(frame, "Face Detected", (x, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Convert for display
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (480, 360))  # Resize for UI
            img = ImageTk.PhotoImage(Image.fromarray(frame))
            self.cam_label.configure(image=img)
            self.cam_label.image = img
            
        self.window.after(15, self.update_camera)
    
    def capture_images(self):
        name = self.name_entry.get().strip()
        if not name:
            messagebox.showerror("Error", "Please enter a name")
            return
        
        if self.current_frame is None:
            messagebox.showerror("Error", "No camera feed available")
            return
            
        self.capture_btn.config(state='disabled')
        self.status_label.config(text="Starting capture...")
        self.progress['value'] = 0
        
        embeddings = []
        successful_captures = 0
        
        for i in range(5):  # Capture 5 samples
            self.status_label.config(text=f"Capturing image {i+1}/5...")
            self.window.update()
            
            # Wait a moment between captures
            time.sleep(1)
            
            ret, frame = self.cap.read()
            if ret:
                faces = detect_faces(frame)
                
                if faces:
                    # Use the largest face detected
                    largest_face = max(faces, key=lambda face: face[2] * face[3])
                    x, y, w, h = largest_face
                    
                    # Extract face with some padding
                    padding = 20
                    y1 = max(0, y - padding)
                    y2 = min(frame.shape[0], y + h + padding)
                    x1 = max(0, x - padding)
                    x2 = min(frame.shape[1], x + w + padding)
                    
                    face_img = frame[y1:y2, x1:x2]
                    
                    # Skip if face is too small
                    if face_img.shape[0] < 50 or face_img.shape[1] < 50:
                        self.status_label.config(text=f"Face too small in image {i+1}, retrying...")
                        continue
                    
                    embedding = self.facenet.get_embedding(face_img)
                    
                    if embedding is not None:
                        embeddings.append(embedding)
                        successful_captures += 1
                        self.status_label.config(text=f"Successfully captured {successful_captures}/5")
                    else:
                        self.status_label.config(text=f"Failed to process image {i+1}")
                else:
                    self.status_label.config(text=f"No face detected in image {i+1}")
            
            # Update progress bar
            self.progress['value'] = ((i + 1) / 5) * 100
            self.window.update()
        
        if successful_captures >= 3:  # Require at least 3 successful captures
            try:
                self.es.store_embeddings(name, embeddings)
                self.status_label.config(text=f"Enrollment successful! ({successful_captures} images captured)")
                messagebox.showinfo("Success", f"Successfully enrolled {name}!")
                self.name_entry.delete(0, tk.END)  # Clear the name field
            except Exception as e:
                self.status_label.config(text="Error storing embeddings")
                messagebox.showerror("Error", f"Failed to store embeddings: {str(e)}")
        else:
            self.status_label.config(text=f"Enrollment failed - only {successful_captures} valid captures")
            messagebox.showerror("Error", 
                               f"Insufficient face captures ({successful_captures}/5). "
                               "Please ensure your face is clearly visible and try again.")
        
        self.progress['value'] = 0
        self.capture_btn.config(state='normal')
    
    def cleanup(self):
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    app = EnrollmentApp()
    try:
        app.window.mainloop()
    finally:
        app.cleanup()