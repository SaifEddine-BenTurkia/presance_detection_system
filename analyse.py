from sklearn.datasets import fetch_lfw_people
import matplotlib.pyplot as plt
import numpy as np

# Path to your local LFW dataset directory (where the dataset is stored)
local_lfw_path = 'lfw-deepfunneled\lfw-deepfunneled'  # <-- Change this to your local path

# Load the LFW-deepfunneled dataset from local storage without downloading
lfw_dataset = fetch_lfw_people(
    min_faces_per_person=0,
    resize=1.0,
    download_if_missing=False,  # Do NOT download if missing
    data_home=local_lfw_path
)

# Basic dataset info
print(f"Total images: {lfw_dataset.images.shape[0]}")
print(f"Image shape: {lfw_dataset.images.shape[1:]} (Height x Width)")
print(f"Number of individuals: {len(lfw_dataset.target_names)}")

# Count images per person
unique, counts = np.unique(lfw_dataset.target, return_counts=True)
images_per_person = dict(zip(lfw_dataset.target_names, counts))

# Print top 5 people with the most images
top_people = sorted(images_per_person.items(), key=lambda x: x[1], reverse=True)[:5]
print("\nTop 5 people with most images:")
for name, count in top_people:
    print(f"{name}: {count} images")

# Optional: Show a sample image
plt.imshow(lfw_dataset.images[0], cmap='gray')
plt.title(lfw_dataset.target_names[lfw_dataset.target[0]])
plt.axis('off')
plt.show()
