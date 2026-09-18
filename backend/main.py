from yolo_service import segment_frames
from colmap_service import run_colmap_reconstruction, run_dense_reconstruction
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import os
import shutil
import uuid
import cv2
import statistics


app = FastAPI(
    title="SIH26158 Drone 3D Reconstruction API",
    description="Backend API for Single-Pass Drone Video to Georeferenced 3D Model",
    version="1.0.0"
)


# =========================
# FOLDERS
# =========================

UPLOAD_FOLDER = "uploads"
FRAMES_FOLDER = "frames"
PROCESSED_FOLDER = "processed_frames"
RECONSTRUCTION_FOLDER = "reconstruction"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(FRAMES_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)
os.makedirs(RECONSTRUCTION_FOLDER, exist_ok=True)


# Temporary job storage
jobs = {}


# =========================
# HOME
# =========================

@app.get("/")
def home():
    return {
        "message": "Drone 3D Reconstruction Backend is running!"
    }


# =========================
# HEALTH CHECK
# =========================

@app.get("/health")
def health():
    return {
        "status": "OK"
    }


# =========================
# UPLOAD VIDEO
# =========================

@app.post("/upload")
async def upload_video(file: UploadFile = File(...)):

    # Check file type
    if not file.filename.lower().endswith(".mp4"):
        return {
            "error": "Only MP4 video files are allowed"
        }

    # Create Job ID
    job_id = str(uuid.uuid4())

    # Save video
    file_path = os.path.join(
        UPLOAD_FOLDER,
        file.filename
    )

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Store job information
    jobs[job_id] = {
        "job_id": job_id,
        "filename": file.filename,
        "path": file_path,
        "status": "uploaded",
        "progress": 0
    }

    return {
        "message": "Video uploaded successfully",
        "job_id": job_id,
        "filename": file.filename,
        "status": "uploaded",
        "progress": 0
    }


# =========================
# GET JOB STATUS
# =========================

@app.get("/jobs/{job_id}")
def get_job_status(job_id: str):

    if job_id not in jobs:
        return {
            "error": "Job not found"
        }

    return jobs[job_id]


# =========================
# EXTRACT FRAMES
# =========================

@app.post("/jobs/{job_id}/start")
def start_processing(job_id: str):

    # Check whether job exists
    if job_id not in jobs:
        return {
            "error": "Job not found"
        }

    job = jobs[job_id]

    video_path = job["path"]

    # Update status
    job["status"] = "processing"
    job["progress"] = 10

    # Open video
    video = cv2.VideoCapture(video_path)

    if not video.isOpened():

        job["status"] = "failed"

        return {
            "error": "Could not open video"
        }

    # Get video information
    total_frames = int(
        video.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    fps = video.get(
        cv2.CAP_PROP_FPS
    )

    # Create folder for this job's frames
    job_frames_folder = os.path.join(
        FRAMES_FOLDER,
        job_id
    )

    os.makedirs(
        job_frames_folder,
        exist_ok=True
    )

    frame_number = 0
    saved_frames = 0

    # Save every 30th frame
    frame_interval = 30

    while True:

        success, frame = video.read()

        if not success:
            break

        if frame_number % frame_interval == 0:

            frame_path = os.path.join(
                job_frames_folder,
                f"frame_{saved_frames:05d}.jpg"
            )

            cv2.imwrite(
                frame_path,
                frame
            )

            saved_frames += 1

        frame_number += 1

    video.release()

    # Update job
    job["status"] = "completed"
    job["progress"] = 100
    job["total_frames"] = total_frames
    job["fps"] = fps
    job["frames_extracted"] = saved_frames
    job["frames_path"] = job_frames_folder

    return {
        "message": "Video processing completed",
        "job_id": job_id,
        "status": "completed",
        "progress": 100,
        "total_frames": total_frames,
        "fps": fps,
        "frames_extracted": saved_frames,
        "frames_path": job_frames_folder
    }


# =========================
# PREPROCESS FRAMES
# =========================

@app.post("/jobs/{job_id}/preprocess")
def preprocess_frames(job_id: str):

    # Check whether job exists
    if job_id not in jobs:

        return {
            "error": "Job not found"
        }

    job = jobs[job_id]

    # Check whether frames were extracted
    if "frames_path" not in job:

        return {
            "error": "Frames have not been extracted yet. Run /start first."
        }

    input_folder = job["frames_path"]

    # Create output folder
    output_folder = os.path.join(
        PROCESSED_FOLDER,
        job_id
    )

    os.makedirs(
        output_folder,
        exist_ok=True
    )

    job["status"] = "preprocessing"
    job["progress"] = 10

    # Store scores
    scores = []

    # =========================
    # STEP 1: CALCULATE SCORES
    # =========================

    for filename in sorted(os.listdir(input_folder)):

        if not filename.lower().endswith(".jpg"):
            continue

        image_path = os.path.join(
            input_folder,
            filename
        )

        image = cv2.imread(image_path)

        if image is None:
            continue

        # Convert to grayscale
        gray = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY
        )

        # Calculate sharpness
        blur_score = cv2.Laplacian(
            gray,
            cv2.CV_64F
        ).var()

        scores.append(
            (filename, blur_score)
        )

    # No frames found
    if not scores:

        job["status"] = "failed"

        return {
            "error": "No valid frames found"
        }

    # =========================
    # STEP 2: MEDIAN
    # =========================

    all_scores = [
        score
        for filename, score in scores
    ]

    median_score = statistics.median(
        all_scores
    )

    # =========================
    # STEP 3: THRESHOLD
    # =========================

    threshold = median_score * 0.70

    # =========================
    # STEP 4: SELECT FRAMES
    # =========================

    kept_frames = []
    rejected_frames = []

    for filename, score in scores:

        if score >= threshold:

            source_path = os.path.join(
                input_folder,
                filename
            )

            destination_path = os.path.join(
                output_folder,
                filename
            )

            shutil.copy2(
                source_path,
                destination_path
            )

            kept_frames.append(
                filename
            )

        else:

            rejected_frames.append(
                filename
            )

    # =========================
    # UPDATE JOB
    # =========================

    job["status"] = "preprocessing_completed"
    job["progress"] = 100

    job["median_blur_score"] = median_score
    job["blur_threshold"] = threshold
    job["frames_checked"] = len(scores)
    job["frames_kept"] = len(kept_frames)
    job["frames_rejected"] = len(rejected_frames)
    job["processed_frames_path"] = output_folder

    return {

        "message": "Frame preprocessing completed",

        "job_id": job_id,

        "status": "preprocessing_completed",

        "progress": 100,

        "frames_checked": len(scores),

        "frames_kept": len(kept_frames),

        "frames_rejected": len(rejected_frames),

        "median_blur_score": median_score,

        "blur_threshold": threshold,

        "processed_frames_path": output_folder
    }
@app.post("/jobs/{job_id}/segment")
def segment_job(job_id: str):

    if job_id not in jobs:
        return {"error": "Job not found"}

    job = jobs[job_id]

    if "processed_frames_path" not in job:
        return {
            "error": "Preprocessing has not been completed. Run /preprocess first."
        }

    input_folder = job["processed_frames_path"]

    output_folder = os.path.join(
        "segmentation",
        job_id
    )

    os.makedirs(output_folder, exist_ok=True)

    job["status"] = "segmentation_processing"
    job["progress"] = 10

    try:

        result = segment_frames(
            input_folder,
            output_folder
        )

        job["status"] = "segmentation_completed"
        job["progress"] = 100

        job["status"] = "segmentation_completed"
        job["progress"] = 100
        job["segmentation_path"] = output_folder
        job["masks_path"] = result["masks_folder"]
        job["masked_preview_path"] = result["preview_folder"]
        job["colmap_images_path"] = result["colmap_images_folder"]
        job["detections_log"] = result["detections_log"]
        job["frames_segmented"] = result["frames_processed"]

        return {
            "message": "YOLO segmentation completed",
            "job_id": job_id,
            "status": "segmentation_completed",
            "progress": 100,
            **result
        }

    except Exception as e:

        job["status"] = "segmentation_failed"

        return {
            "error": str(e),
            "job_id": job_id
        }
@app.post("/jobs/{job_id}/reconstruct")
def reconstruct_job(job_id: str):

    if job_id not in jobs:
        return {"error": "Job not found"}

    job = jobs[job_id]

    if "colmap_images_path" not in job:
        return {
            "error": "COLMAP images are not available. Run /segment first."
        }

    image_folder = job["colmap_images_path"]

    workspace_folder = os.path.join(
        "reconstruction",
        job_id
    )

    job["status"] = "reconstruction_processing"
    job["progress"] = 10

    try:

        result = run_colmap_reconstruction(
            image_folder,
            workspace_folder
        )

        job["status"] = "reconstruction_completed"
        job["progress"] = 100
        job["reconstruction_path"] = workspace_folder
        job["colmap_database"] = result["database_path"]
        job["sparse_model_path"] = result["sparse_path"]

        return {
            "message": "COLMAP reconstruction completed",
            "job_id": job_id,
            "status": "reconstruction_completed",
            "progress": 100,
            "reconstruction_path": workspace_folder,
            "database_path": result["database_path"],
            "sparse_model_path": result["sparse_path"]
        }

    except Exception as e:

        job["status"] = "reconstruction_failed"

        return {
            "error": str(e),
            "job_id": job_id,
            "status": "reconstruction_failed"
        }
# ============================================================
# DENSE RECONSTRUCTION
# ============================================================

@app.post("/jobs/{job_id}/dense")
def dense_reconstruction(job_id: str):

    if job_id not in jobs:
        image_folder = os.path.join("segmentation", job_id, "colmap_images")
        workspace_folder = os.path.join(RECONSTRUCTION_FOLDER, job_id)
        sparse_folder = os.path.join(workspace_folder, "sparse", "0")

        if not os.path.isdir(image_folder):
            return JSONResponse(
                status_code=404,
                content={"error": "COLMAP images not found for this job."}
            )

        if not os.path.isdir(sparse_folder):
            return JSONResponse(
                status_code=404,
                content={"error": "Sparse reconstruction not found for this job."}
            )

        jobs[job_id] = {
            "job_id": job_id,
            "status": "reconstruction_completed",
            "progress": 100,
            "colmap_images_path": image_folder,
            "sparse_model_path": os.path.join(workspace_folder, "sparse")
        }

    job = jobs[job_id]

    # Check sparse reconstruction
    if "sparse_model_path" not in job:
        return JSONResponse(
            status_code=400,
            content={
                "error": "Sparse reconstruction is not available. "
                         "Run /reconstruct first."
            }
        )

    image_folder = job.get("colmap_images_path")

    if not image_folder:
        return JSONResponse(
            status_code=400,
            content={
                "error": "COLMAP images are not available. "
                         "Run /segment first."
            }
        )

    workspace_folder = os.path.join(
        RECONSTRUCTION_FOLDER,
        job_id
    )

    try:

        job["status"] = "dense_reconstruction_started"
        job["progress"] = 0

        result = run_dense_reconstruction(
            image_folder,
            workspace_folder
        )

        job["dense_path"] = result["dense_path"]
        job["fused_point_cloud"] = result["fused_point_cloud"]

        job["status"] = "dense_reconstruction_completed"
        job["progress"] = 100

        return {
            "message": "Dense reconstruction completed",
            "job_id": job_id,
            "status": job["status"],
            "progress": 100,
            "dense_path": result["dense_path"],
            "fused_point_cloud": result["fused_point_cloud"]
        }

    except Exception as e:

        job["status"] = "dense_reconstruction_failed"

        return JSONResponse(
            status_code=500,
            content={
                "error": str(e),
                "job_id": job_id,
                "status": job["status"]
            }
        )