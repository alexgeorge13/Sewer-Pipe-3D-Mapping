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

Typical 3D mapping pipelines scale quadratically or cubically relative to tracked points. This framework scales linearly $(O(M \times N))$ by utilizing a specialized Extended Kalman Filter (EKF) layout tailored to pipe bounds, allowing the mapping backend to run in real-time on standard commercial hardware.

### 1. Weakly Supervised Semantic Perception
To minimize manual training costs, the pipeline avoids pixel-level or bounding-box annotations by utilizing a Multi-type Fully Convolutional Anomaly Detection (**MultiTypeFCDD**) network. 
* **Anomaly Mapping:** Input frames are evaluated by a fully convolutional network across $M$ target classes via a pseudo-Huber transformation:
  <br><img src="https://latex.codecogs.com/svg.image?A_k(X_t)%20=%20%5Csqrt%7B%5Cphi(X_t;W)^2%20+%201%7D%20-%201" title="A_k(X_t) = \sqrt{\phi(X_t;W)^2 + 1} - 1" /><br>
* **Centroid Detections:** The resulting heatmaps are upsampled. Discrete 2D coordinate observations $z_j = [u, v]^\top$ are extracted dynamically by computing the centroids of connected pixel components that breach a strict anomaly probability threshold ($A'_k > \tau$).

### 2. Dynamic Principal Point & Focus of Expansion (FOE)
Camera positioning inside moving crawlers drifts continuously due to loose tether tensions and floor unevenness. To eliminate full camera pose estimation requirements, the system dynamically calculates the pipe's vanishing point (FOE) using optical flow fields:
* Every tracked pixel moving at a specific velocity enforces a linear constraint: $-Vy,i c_{x,t} + Vx,i c_{y,t} = Vx,i v_i - Vy,i u_i$.
* An overdetermined system $Ac = b$ is constructed and evaluated using **RANSAC** to isolate an outlier-free inlier subset. The optimized principal point is continually resolved via:
  <br><img src="https://latex.codecogs.com/svg.image?c^*%20=%20%5Carg%5Cmin_c%20%5C%7CA_%7Bin%7Dc%20-%20b_%7Bin%7D%5C%7C_2^2" title="c^* = \arg\min_c \|A_{in}c - b_{in}\|_2^2" /><br>

### 3. Cylinder-to-Image Projection Model
By relying on the physical geometry of a pipe with a known radius $R$, a static landmark in the 3D global world frame maps to: $P_w = [R \cos \theta_i, R \sin \theta_i, l_{x,i}]^\top$. Assuming longitudinal camera alignment where relative depth is defined as $\Delta x = l_{x,i} - x_r$ (provided via tether odometry), the highly efficient 3D-to-2D projection reduces to:
<br><img src="https://latex.codecogs.com/svg.image?%5Chat%7Bz%7D_t%20=%20%5Cbegin%7Bbmatrix%7D%20c_%7Bx,t%7D%20+%20%5Cfrac%7Bf_x%20R%20%5Ccos%20%5Ctheta_i%7D%7Bl_%7Bx,i%7D%20-%20x_r%7D%20%5C%5C%20c_%7By,t%7D%20+%20%5Cfrac%7Bf_y%20R%20%5Csin%20%5Ctheta_i%7D%7Bl_%7Bx,i%7D%20-%20x_r%7D%20%5Cend%7Bbmatrix%7D" title="\hat{z}_t = \begin{bmatrix} c_{x,t} + \frac{f_x R \cos \theta_i}{l_{x,i} - x_r} \\ c_{y,t} + \frac{f_y R \sin \theta_i}{l_{x,i} - x_r} \end{bmatrix}" /><br>

### 4. Structure-Only EKF Tracking & Linearization
The backend instantiates a bank of independent, stationary Extended Kalman Filters (EKFs) for each landmark. To map spatial uncertainty from the 3D cylindrical canvas into the 2D image plane, the projection function is linearized at each step via the Jacobian matrix $H_{i,t}$:
<br><img src="https://latex.codecogs.com/svg.image?H_%7Bi,t%7D%20=%20%5Cbegin%7Bbmatrix%7D%20-%5Cfrac%7Bf_x%20R%20%5Ccos%20%5Ctheta_i%7D%7B%5CDelta%20x^2%7D%20&%20-%5Cfrac%7Bf_x%20R%20%5Csin%20%5Ctheta_i%7D%7B%5CDelta%20x%7D%20%5C%5C%20-%5Cfrac%7Bf_y%20R%20%5Csin%20%5Ctheta_i%7D%7B%5CDelta%20x^2%7D%20&%20%5Cfrac%7Bf_y%20R%20%5Ccos%20%5Ctheta_i%7D%7B%5CDelta%20x%7D%20%5Cend%7Bbmatrix%7D" title="H_{i,t} = \begin{bmatrix} -\frac{f_x R \cos \theta_i}{\Delta x^2} & -\frac{f_x R \sin \theta_i}{\Delta x} \\ -\frac{f_y R \sin \theta_i}{\Delta x^2} & \frac{f_y R \cos \theta_i}{\Delta x} \end{bmatrix}" /><br>

### 5. Data Association & Geometric Back-Projection
* **GNN Assignment:** Incoming semantic detections $z_j$ are coupled with existing map tracks by minimizing the Mahalanobis distance. Assignments that clear a validation gate ($\gamma = 9.21$) are globally optimized using the **Duff-Koster linear assignment algorithm**.
* **Track Initialisation:** Any unassociated detections are back-projected directly into the 3D map space. Using the current principal point $c_t$, the system computes the radial displacement $\rho$:
  <br><img src="https://latex.codecogs.com/svg.image?%5Crho%20=%20%5Csqrt%7B%5Cfrac%7B(u%20-%20c_%7Bx,t%7D)^2%7D%7Bf_x^2%7D%20+%20%5Cfrac%7B(v%20-%20c_%7By,t%7D)^2%7D%7Bf_y^2%7D%7D" title="\rho = \sqrt{\frac{(u - c_{x,t})^2}{f_x^2} + \frac{(v - c_{y,t})^2}{f_y^2}}" /><br>
  New landmark tracks are initialized at:
  <br><img src="https://latex.codecogs.com/svg.image?%5Chat%7Bl%7D_x%20=%20x_r%20+%20%5Cfrac%7BR%7D%7B%5Crho%7D%20%5Cquad%20%5Ctext%7Band%7D%20%5Cquad%20%5Chat%7B%5Ctheta%7D%20=%20%5Ctext%7Batan2%7D%5Cleft(%5Cfrac%7Bv%20-%20c_%7By,t%7D%7D%7Bf_y%7D,%20%5Cfrac%7Bu%20-%20c_%7Bx,t%7D%7D%7Bf_x%7D%5Cright)" title="\hat{l}_x = x_r + \frac{R}{\rho} \quad \text{and} \quad \hat{\theta} = \text{atan2}\left(\frac{v - c_{y,t}}{f_y}, \frac{u - c_{x,t}}{f_x}\right)" /><br>
  Tracks are held in candidate storage and are only promoted to the global map after $N_{min}$ persistent detections to filter out noise.
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
4. **Data Balancing & Splitting:** For balanced training, extract a random subset of **1000 images per class** and save them in a root folder named `WRc_Dataset`. The script automatically utilizes a **70/10/20 split** for training, validation, and testing.
5. **Outputs:** Running `train_multiTypeFCDD.m` evaluates performance metrics, displays a heatmap visualization of anomalies for sample test data, and saves the weights/anomaly thresholds to `trainedMultiTypeFCDDNet.mat`.

### 2. The Video Dataset
The inspection video stream is sourced from the University of Sheffield's ORDA repository:
1. Download the data from the ORDA dataset link: [https://orda.shef.ac.uk/articles/dataset/Visual_Odometry_for_Robot_Localisation_in_Feature-Sparse_Sewer_Pipes_Using_Joint_and_Manhole_Detections_--_Data/21198070](https://orda.shef.ac.uk/articles/dataset/Visual_Odometry_for_Robot_Localisation_in_Feature-Sparse_Sewer_Pipes_Using_Joint_and_Manhole_Detections_--_Data/21198070).
2. Select and download the video **`Pipe1.avi`**.
3. **Important:** It is highly recommended to convert this video file from `.avi` to an **`.mp4`** format before loading. Ensure your final filename match matches the script execution target (`Pipe1.mp4`).

---
## Execution Manual

Verify that your file tree matches the workspace architecture outlined below:

```text
Sewer-Pipe-3D-Mapping/
├── WRc_Dataset/                    
├── main_EKF_mapping.m             
├── setup_config.m                  
├── get_focus_of_expansion.m      
├── resizeImage.m                  
├── perform_gnn_association.m       
├── initialize_new_tracks.m        
├── update_existing_tracks.m        
├── merge_redundant_tracks.m       
├── train_multiTypeFCDD.m             
├── customAnomalyScore.m               
├── augmentImageData.m                 
├── balancedMultiClassDatastore.m     
├── sewerPipe_groundTruthAnomaly.csv   
├── trainedMultiTypeFCDDNet.mat       
└── sewerPipeVideo.mp4                 
```

Once that is sorted, in the MATLAB Command Window, execute `main_EKF_mapping`.

This script serves as the main orchestrator of the pipeline, unifying the deep learning anomaly detector with a robust backend bank of Extended Kalman Filters. By continuously calculating the pipe's focus of expansion via optical flow, it dynamically corrects for camera movement to ensure accurate 3D spatial mapping. The entire framework runs natively within a single, unified execution loop without requiring external tracking dependencies.

Upon completion, the system exports the real-time perspective visualization directly as a high-resolution video file named `3D_Mapping_Husky_Rover.mp4`. Additionally, the final global 3D pipe reconstruction, complete with mapped defect landmarks and ground-truth verification comparisons, is saved locally as a file titled `Final_Mapping.png`.
