# Sewer Pipe 3D Mapping

An Extended Kalman Filter (EKF) mapping pipeline designed for tether-anchored robotic inspection in cylindrical environments (sewer pipes). This repository features a 3D reconstruction dashboard tracking semantic features over time alongside a simulated Clearpath Husky rover model.

## System Architecture & Workflow

The pipeline utilizes multi-class neural segmentation and dense optical feature tracking to identify pipeline defects, projects them recursively using a monocular camera model, and refines their structural locations over time using a specialized EKF layout.

1. **Inference & Focus of Expansion (FoE):** Extracts semantic blobs and visual centering tracking parameters.
2. **Data Association:** Employs a Global Nearest Neighbor (GNN) paradigm to match active detections to established anomaly tracks.
3. **Recursive EKF Correction:** Refines distance and angular orientation parameters.
4. **Post-Processing Heuristics:** Deduplicates spatial tracking clusters and prunes false-positives utilizing physical plumbing priors (e.g., removing "sedimentary deposits" detected on pipe ceilings).

## 🛠️ Repository File Mapping

* `main_EKF_mapping.m` — Main execution loop and dashboard pipeline.
* `setup_config.m` — Defines global spatial covariance, noise matrices, and cylinder geometric bounds.
* `initialize_new_tracks.m` / `update_existing_tracks.m` — Core EKF state vector processing engines.
* `perform_gnn_association.m` — Feature matching matrix logic.
* `merge_redundant_tracks.m` — Real-time tracking deduplication routine.
* `get_focus_of_expansion.m` / `resizeImage.m` — Feature extraction and video spatial helpers.
* `train_multiTypeFCDD.m` / `customAnomalyScore.m` — Neural network training frameworks.

## 📋 Prerequisites & Dataset Setup

Follow the steps below to gather the required datasets and prepare them for the pipeline:

### 1. The WRc Image Dataset
To access the image training data, you must download the WRc dataset:
1. Create an account on the Spring website: [https://spring-innovation.co.uk/](https://spring-innovation.co.uk/).
2. Search for **"AI and Sewer case study"** and locate the download link inside the blog post.
3. The dataset folders are labeled using **MSCC5 defect code standards**. This pipeline maps those codes into **5 distinct classes** as follows:
   * **Connection:** `CN`
   * **Deposit:** `DEC`, `DEE`, `DEF`, `DEG_1`, `DEG_2`, `DER`, `DES`
   * **Displaced Joint:** `JDL`, `JDM_1`, `JDM_2`
   * **Fracture:** `FC`, `FL`, `FM`, `FS`
   * **Roots:** `RF`, `RM_1`, `RM_2`, `RT`
4. **Data Balancing & Splitting:** For balanced training, extract a random subset of **1000 images per class** and save them in a root folder named `WRc_Dataset/`. The script automatically utilizes a **70/10/20 split** for training, validation, and testing.
5. **Outputs:** Running `train_multiTypeFCDD.m` evaluates performance metrics, displays a heatmap visualization of anomalies for sample test data, and saves the weights/anomaly thresholds to `trainedMultiTypeFCDDNet.mat`.

### 2. The Video Dataset
The inspection video stream is sourced from the University of Sheffield's ORDA repository:
1. Download the data from the ORDA dataset link: [https://orda.shef.ac.uk/articles/dataset/Visual_Odometry_for_Robot_Localisation_in_Feature-Sparse_Sewer_Pipes_Using_Joint_and_Manhole_Detections_--_Data/21198070](https://orda.shef.ac.uk/articles/dataset/Visual_Odometry_for_Robot_Localisation_in_Feature-Sparse_Sewer_Pipes_Using_Joint_and_Manhole_Detections_--_Data/21198070).
2. Select and download the video **`Pipe1.avi`**.
3. **Important:** It is highly recommended to convert this video file from `.avi` to an **`.mp4`** format before loading. Ensure your final filename match matches the script execution target (`sewerPipeVideo.mp4`).

### 3. Validation Markers
Ensure `sewerPipe_groundTruthAnomaly.csv` is present in the root folder to accurately render the final comparative evaluation figures.

---
Run `main_EKF_mapping.m` inside MATLAB to begin processing.
