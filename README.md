# Sewer Pipe 3D Mapping

This repository contains the official MATLAB implementation of a lightweight 3D semantic mapping pipeline designed for tether-anchored robotic inspection in cylindrical environments like sewer pipes. By leveraging a geometric cylindrical prior, the framework overcomes standard monocular camera limitations such as depth ambiguity, scale drift, and low-parallax processing of distant features without requiring expensive, heavy sensor suites (like LiDAR or IMUs) or intensive 3D scene reconstruction engines.

If you make use of this code or the findings in your academic work, please cite the original conference paper:

#### BibTeX
```bibtex
@inproceedings{george2026lightweight,
  title={Lightweight semantic 3D mapping in sewer pipes leveraging cylindrical geometry},
  author={George, A and Mihaylova, L and Anderson, SR},
  booktitle={Proceedings of the 12th 2026 International Conference on Control, Decision and Information Technologies (CoDIT 2026)},
  year={2026},
  organization={Institute of Electrical and Electronics Engineers (IEEE)}
}
```

#### Standard Text
> George, A., Mihaylova, L., & Anderson, S. R. (2026, April). Lightweight semantic 3D mapping in sewer pipes leveraging cylindrical geometry. In Proceedings of the 12th 2026 International Conference on Control, Decision and Information Technologies (CoDIT 2026). Institute of Electrical and Electronics Engineers (IEEE).

---

## System Architecture & Core Concepts

Typical 3D mapping pipelines scale quadratically or cubically relative to tracked points. This framework scales linearly ($O(M \times N)$) by utilizing a specialized Extended Kalman Filter (EKF) layout tailored to pipe bounds, allowing the mapping backend to run in real-time on standard commercial hardware.

### 1. Weakly Supervised Semantic Perception
Instead of relying on dense, expensive bounding-box or pixel-level annotations, the front-end features a Multi-type Fully Convolutional Anomaly Detection (**MultiTypeFCDD**) network. It maps image-level binary labels to high-resolution anomaly probability heatmaps. Discrete landmark observations are extracted dynamically by taking the centroids of connected components where pixel intensity exceeds a designated threshold ($\tau$).

### 2. Dynamic Principal Point (FOE) Estimation
Standard algorithms assume a static camera centerline. Because physical inspection crawlers shift and drift due to uneven floor surfaces and tether pulling forces, this pipeline uses optical flow fields combined with RANSAC to calculate the dynamic **Focus of Expansion (FOE)**. This tracks the vanishing point of the pipe in real-time.

### 3. Structure-Only EKF Tracking
Instead of managing a massive, computationally expensive 6-DOF state representation, the pipeline handles landmarks as static 2D coordinates within a cylindrical coordinate frame:
$$x_i = [l_{x,i}, \theta_i]^\top$$
Where $l_{x,i}$ is the absolute axial distance along the pipe baseline, and $\theta_i$ represents the continuous angular orientation along the circumference. The longitudinal position of the rover ($x_r$) is directly fed into the filter via a calibrated tether odometer.

---

## Prerequisites & Dataset Setup

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
3. **Important:** It is highly recommended to convert this video file from `.avi` to an **`.mp4`** format before loading. Ensure your final filename match matches the script execution target (`Pipe1.mp4`).

### 3. Validation Markers
Ensure `sewerPipe_groundTruthAnomaly.csv` is present in the root folder to accurately render the final comparative evaluation figures.

---
## Execution Manual

Once your file tree matches the workspace architecture outlined below, open your environment in MATLAB and call the core loop:

```text
Sewer-Pipe-3D-Mapping/
├── WRc_Dataset/                    <-- 5,000 balanced sample image folders
├── main_EKF_mapping.m
├── setup_config.m
├── sewerPipe_groundTruthAnomaly.csv
├── trainedMultiTypeFCDDNet.mat     <-- Exported weights file
└── sewerPipeVideo.mp4              <-- Transcoded inspection target asset
```
In the MATLAB Command Window, execute:
main_EKF_mapping
