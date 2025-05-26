# Number of persons with one image

'''import os

# ---- CONFIGURATION ----
dataset_dir = "lfw-deepfunneled/lfw-deepfunneled"  # Change if needed
output_file = "excluded_persons.txt"

# ---- COLLECT PERSONS WITH ONLY ONE IMAGE ----
excluded_persons = []

for person in os.listdir(dataset_dir):
    person_dir = os.path.join(dataset_dir, person)
    if os.path.isdir(person_dir):
        image_files = [
            img for img in os.listdir(person_dir)
            if img.lower().endswith(('.jpg', '.jpeg', '.png'))
        ]
        if len(image_files) == 1:
            excluded_persons.append(person)

# ---- SAVE RESULTS ----
with open(output_file, "w") as f:
    for person in excluded_persons:
        f.write(f"{person}\n")

# ---- REPORT ----
print(f"Total persons with only 1 image: {len(excluded_persons)}")
print(f"Excluded person names saved to: {output_file}")


'''
import os

dataset_dir = "lfw-deepfunneled/lfw-deepfunneled"  # Update if needed

total_images = 0
person_image_counts = {}

for person in os.listdir(dataset_dir):
    person_dir = os.path.join(dataset_dir, person)
    if os.path.isdir(person_dir):
        images = [img for img in os.listdir(person_dir) if os.path.isfile(os.path.join(person_dir, img))]
        total_images += len(images)
        person_image_counts[person] = len(images)

print(f"✅ Total images in dataset: {total_images}")
print(f"👥 Total persons: {len(person_image_counts)}")
