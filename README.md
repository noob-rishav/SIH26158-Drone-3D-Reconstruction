# SIH26158 - Single-Pass Drone Video to Georeferenced 3D Model

Smart India Hackathon 2026

## Overview

This project focuses on generating a georeferenced 3D reconstruction from a single-pass drone video, using computer vision and photogrammetry techniques.

The main objective is to build an end-to-end reconstruction pipeline that reduces dependency on traditional Ground Control Points (GCPs) by relying on GPS and IMU data for metric-scale georeferencing, while remaining robust to dynamic objects, motion blur, and limited viewing angles.

## Problem

* Single-pass flights provide limited overlap and viewing angles
* Dynamic objects, blur, and illumination changes degrade reconstruction quality
* GPS alone does not produce a complete, accurate 3D model
* GCP-heavy workflows increase field survey effort
* Raw reconstruction output is difficult for non-technical users to interpret

## Workflow


Drone Video + GPS/Metadata
     |
     v
Ingestion & Preprocessing (keyframe selection, blur rejection, GPS/IMU sync)
     |
     v
Dynamic Object Masking (YOLOv8-seg)
     |
     v
Structure from Motion (GLOMAP / COLMAP)
     |
     v
Georeferencing / Metric Scale (GPS to ENU)
     |
     v
Dense Reconstruction (MVS)
     |
     v
Meshing & Texturing
     |
     v
Semantic Classification (terrain, buildings, roads, vegetation)
     |
     v
Cleanup & Occlusion Handling
     |
     v
Delivery (OBJ/GLTF, PLY/LAS, KML/KMZ, Web Viewer)

## Technologies

* Python
* OpenCV
* YOLOv8-seg
* COLMAP / GLOMAP (Structure from Motion)
* Open3D
* pyproj / pymap3d (georeferencing)
* FastAPI
* Three.js
* Blender Python API
* Kaggle / Colab Notebooks

## Repository Structure

SIH26158-Drone-3D-Reconstruction/
|
├── docs/
├── notebooks/
├── src/
└── README.md

* `docs/` - Project documentation, architecture diagrams, and reports
* `notebooks/` - Experiments and development notebooks (Colab/Kaggle)
* `src/` - Pipeline source code

## Getting Started

Clone the repository:

```bash
git clone https://github.com/noob-rishav/SIH26158-Drone-3D-Reconstruction.git
cd SIH26158-Drone-3D-Reconstruction
```

The main experiments are available in the `notebooks/` directory.

## Team

Team Trinetra

## Project Status

This project is currently under development as part of Smart India Hackathon 2026. Pipeline architecture and team roles are finalized; prototype implementation is in progress.