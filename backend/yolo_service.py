from ultralytics import YOLO
import cv2
import os
import json


# Load YOLOv8 segmentation model
model = YOLO("yolov8n-seg.pt")


# Objects that can move and interfere with 3D reconstruction
DYNAMIC_CLASSES = {
    "person",
    "car",
    "truck",
    "bus",
    "motorcycle",
    "bicycle",
    "dog",
    "cat",
    "horse",
    "bird",
    "cow",
    "sheep"
}


def segment_frames(input_folder, output_folder):

    masks_folder = os.path.join(output_folder, "masks")
    preview_folder = os.path.join(output_folder, "masked_preview")
    colmap_folder = os.path.join(output_folder, "colmap_images")

    os.makedirs(masks_folder, exist_ok=True)
    os.makedirs(preview_folder, exist_ok=True)
    os.makedirs(colmap_folder, exist_ok=True)

    detections = []

    frame_files = sorted(
        f for f in os.listdir(input_folder)
        if f.lower().endswith(".jpg")
    )

    for filename in frame_files:

        image_path = os.path.join(input_folder, filename)

        image = cv2.imread(image_path)

        if image is None:
            continue

        # Keep a copy for the COLMAP image
        colmap_image = image.copy()

        results = model(image, verbose=False)

        frame_detections = []

        for result in results:

            if result.masks is None:
                continue

            masks = result.masks.data
            classes = result.boxes.cls

            for i in range(len(masks)):

                class_id = int(classes[i])
                class_name = model.names[class_id]

                # Only remove dynamic objects
                if class_name not in DYNAMIC_CLASSES:
                    continue

                mask = masks[i].cpu().numpy()

                mask = cv2.resize(
                    mask,
                    (image.shape[1], image.shape[0])
                )

                binary_mask = (mask > 0.5).astype("uint8") * 255

                # Save individual mask
                mask_filename = (
                    os.path.splitext(filename)[0]
                    + f"_{class_name}_{i}.png"
                )

                mask_path = os.path.join(
                    masks_folder,
                    mask_filename
                )

                cv2.imwrite(mask_path, binary_mask)

                # Remove object from preview
                mask_image = binary_mask > 0
                colmap_image[mask_image] = 0

                frame_detections.append({
                    "class": class_name,
                    "mask": mask_filename
                })

        # Save masked preview
        preview_path = os.path.join(
            preview_folder,
            filename
        )

        cv2.imwrite(preview_path, colmap_image)

        # Save COLMAP-ready image
        colmap_path = os.path.join(
            colmap_folder,
            filename
        )

        cv2.imwrite(colmap_path, colmap_image)

        detections.append({
            "frame": filename,
            "objects": frame_detections
        })

    # Save detection log
    log_path = os.path.join(
        output_folder,
        "detections_log.json"
    )

    with open(log_path, "w") as f:
        json.dump(detections, f, indent=4)

    return {
        "frames_processed": len(frame_files),
        "masks_folder": masks_folder,
        "preview_folder": preview_folder,
        "colmap_images_folder": colmap_folder,
        "detections_log": log_path
    }