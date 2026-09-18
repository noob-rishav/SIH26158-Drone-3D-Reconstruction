import cv2
import os

frames_folder = "processed_frames/b6e17983-f4b9-466a-ad3d-1b4e9260e616"

files = sorted([
    f for f in os.listdir(frames_folder)
    if f.lower().endswith(".jpg")
])

previous_image = None

for filename in files:

    image_path = os.path.join(
        frames_folder,
        filename
    )

    image = cv2.imread(image_path)

    # Resize for faster comparison
    image = cv2.resize(image, (300, 300))

    if previous_image is not None:

        difference = cv2.absdiff(
            previous_image,
            image
        )

        similarity = 1 - (
            difference.mean() / 255
        )

        similarity_percentage = similarity * 100

        print(
            filename,
            "->",
            round(similarity_percentage, 2),
            "%"
        )

    previous_image = image