import cv2
import os
import statistics

frames_folder = "frames/0b2925a0-45c7-437f-872c-e67f55d8661a"

scores = []

# Calculate blur score for every frame
for filename in sorted(os.listdir(frames_folder)):

    if filename.endswith(".jpg"):

        image_path = os.path.join(frames_folder, filename)

        image = cv2.imread(image_path)

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        blur_score = cv2.Laplacian(
            gray,
            cv2.CV_64F
        ).var()

        scores.append((filename, blur_score))


# Calculate median score
all_scores = [score for filename, score in scores]

median_score = statistics.median(all_scores)

# Set threshold
threshold = median_score * 0.70

print("Median blur score:", median_score)
print("Blur threshold:", threshold)
print()
print("Frame selection:")
print("----------------")

for filename, score in scores:

    if score >= threshold:
        print(filename, "->", round(score, 2), "KEEP")
    else:
        print(filename, "->", round(score, 2), "REJECT")