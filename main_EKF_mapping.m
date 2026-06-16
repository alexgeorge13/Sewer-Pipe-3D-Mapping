% EKF-Based Tether-Anchored Mapping with 3D Rover Model (Husky Integration)
% -------------------------------------------------------------------------
clear all; close all; clc;

% --- Main Configuration ---
cfg = setup_config();
tracks = struct('mu', {}, 'P', {}, 'id', {}, 'color', {}, 'hits', {}, 'misses', {}, 'class', {});
state.next_id = 1;
inferenceSkip = 2; 
targetFrameNum = 3128; % Capture graphics for final figure
startFrame = 2500;
endFrame = 4300; % Start and end frames of the 28-meter segment

% Load Real Video and Odometry
vReader = VideoReader('Pipe1.mp4');
gTruth = readtable('sewerPipe_groundTruthAnomaly.csv');
pipeOdom = gTruth.Distance * 1000; % Raw odometry in mm

% Load Trained Neural Network
load('trainedMultiTypeFCDDNet.mat');
classNames = ["Connection", "Deposit", "Displaced Joint", "Fracture", "Roots"];
numAnomalyClasses = numel(thresholds);

% Setup smoothing history
frame1 = readFrame(vReader);
[imgH, imgW, ~] = size(frame1);
vReader.CurrentTime = 0;
ppX_hist = repmat(imgW/2, 1, cfg.pp_window);
ppY_hist = repmat(imgH/2, 1, cfg.pp_window);

% Setup Output Video
outVideoName = '3D_Mapping_Husky_Rover.mp4';
vWriter = VideoWriter(outVideoName, 'MPEG-4');
vWriter.FrameRate = vReader.FrameRate;
open(vWriter);

% =========================================================================
% SETUP LIVE DASHBOARD FIGURE
% =========================================================================
hFig = figure('Color', 'w', 'Name', '3D Perspective', 'Position', [100, 100, 1000, 500]);
axVideo = subplot(1, 2, 1); hold(axVideo, 'on');
axMap3D = subplot(1, 2, 2);  hold(axMap3D, 'on'); grid(axMap3D, 'on');

% --- Video Side ---
hVideoImg = imshow(zeros(imgH, imgW, 3, 'uint8'), 'Parent', axVideo);
% NEW: Lock the axis limits to the exact image size to prevent shrinking
xlim(axVideo, [1 imgW]); 
ylim(axVideo, [1 imgH]);
hSpiderCenter = plot(axVideo, 0, 0, 'w+', 'MarkerSize', 10, 'LineWidth', 2);
hMeshRings = plot(axVideo, nan, nan, 'Color', [0.7 0.7 0.7], 'LineWidth', 0.5);
title(axVideo, 'CCTV Inspection Video with Overlays');

% --- 3D Reconstruction Side ---
title(axMap3D, '3D Pipe Reconstruction');
view(axMap3D, [-35, 25]); 
axis(axMap3D, 'equal');
zlabel(axMap3D, 'Z (mm)'); ylabel(axMap3D, 'Y (mm)'); xlabel(axMap3D, 'Distance (meters)');

% Generate a 3D Cylinder shell (Treadmill Effect Setup)
ringSpacing = 250; % Spacing between visual rings in mm
x_base = -1000 : ringSpacing : 4500; % Extend slightly past the view window
[Theta, X_grid] = meshgrid(linspace(0, 2*pi, 40), x_base);
cylY = cfg.pipe_R * cos(Theta);
cylZ = cfg.pipe_R * sin(Theta);
hPipeMesh = mesh(axMap3D, X_grid, cylY, cylZ, 'FaceAlpha', 0.05, ...
    'EdgeColor', [0.7 0.7 0.8], 'HandleVisibility', 'off');

% --- 3D ROVER MODEL INTEGRATION ---
hRoverGroup = hgtransform('Parent', axMap3D);
rover = loadrobot("clearpathHusky", "DataFormat", "column");
existingChildren = axMap3D.Children;
show(rover, 'Parent', axMap3D, 'PreservePlot', false);
allChildren = axMap3D.Children;
newChildren = setdiff(allChildren, existingChildren);
for i = 1:length(newChildren)
    if isprop(newChildren(i), 'Parent')
        newChildren(i).Parent = hRoverGroup;
    end
end

S = makehgtform('translate', [0, 0, -cfg.pipe_R/2], 'zrotate', pi, 'scale', 400);
set(hRoverGroup, 'Matrix', S);
view(axMap3D, [-35, 25]); 

% --- OBJECT POOLING ---
maxPool = 50;
hDetPool = gobjects(maxPool, 1);
hMapPoints3D = gobjects(maxPool, 1);
hMapLabels3D = gobjects(maxPool, 1);

% Video Bounding Box and Label Pools
hVideoBBoxes = gobjects(maxPool, 1);
hVideoLabels = gobjects(maxPool, 1);

for i = 1:maxPool
    hDetPool(i) = plot(axVideo, nan, nan, 's', 'MarkerSize', 10, 'LineWidth', 2, 'Visible', 'off');
    hMapPoints3D(i) = scatter3(axMap3D, nan, nan, nan, 100, 'filled', 'MarkerEdgeColor', 'k', 'Visible', 'off');
    hMapLabels3D(i) = text(axMap3D, 0, 0, 0, '', 'FontSize', 9, 'FontWeight', 'bold', 'Visible', 'off');
    
    % Initialize empty rectangles and text for the 2D video
    hVideoBBoxes(i) = rectangle(axVideo, 'Position', [0 0 1 1], 'EdgeColor', 'w', 'LineWidth', 2, 'Visible', 'off');
    hVideoLabels(i) = text(axVideo, 0, 0, '', 'FontSize', 10, 'FontWeight', 'bold', 'Color', 'w', 'Visible', 'off', 'Clipping', 'on');
end

% --- DYNAMIC LEGEND SETUP ---

hLegendDummies = gobjects(size(cfg.baseColors, 1), 1);
for c = 1:size(cfg.baseColors, 1)
    % Plot an invisible point for each class color
    hLegendDummies(c) = scatter3(axMap3D, nan, nan, nan, 100, 'filled', ...
        'MarkerEdgeColor', 'k', 'MarkerFaceColor', cfg.baseColors(c, :));
end

% Create a sleek, dark-themed legend and lock it so it doesn't try to update every frame
legend(axMap3D, hLegendDummies, classNames, 'Location', 'northeast', ...
    'TextColor', 'k', 'Color', 'w', 'AutoUpdate', 'off', 'FontSize', 10, 'FontWeight', 'bold');

% Centre point estimation usng Optic Flow ---
opticFlow = opticalFlowFarneback('NumPyramidLevels', 3, 'PyramidScale', 0.5, ...
    'NumIterations', 3, 'NeighborhoodSize', 5, 'FilterSize', 15);

% =========================================================================
% MAIN LOOP
% =========================================================================
detections = zeros(3, 0); % Initialize as 3x0 matrix to ensure concatenation works
estimated_pp = [imgW/2; imgH/2];
persistentAnomalyMask = zeros(imgH, imgW, 3, 'double');

for frameCount = startFrame:endFrame
    if ~hasFrame(vReader), break; end
    img = read(vReader, frameCount);
    xr_tether = pipeOdom(frameCount);

    % 1. Inference Update
    if mod(frameCount, inferenceSkip) == 0

        % Grayscaling for optic flow computation
        imgGray = rgb2gray(img);

        % Downsample slightly for speed, calculate flow, then scale back
        imgSmallFlow = imresize(imgGray, 0.25);
        flow = estimateFlow(opticFlow, imgSmallFlow);

        [rawX, rawY] = get_focus_of_expansion(flow, imgW, imgH);

        ppX_hist = [ppX_hist(2:end), rawX];
        ppY_hist = [ppY_hist(2:end), rawY];
        estimated_pp = [median(ppX_hist); median(ppY_hist)];

        img_resized = resizeImage(img,detector.Layers(1).InputSize(1:2));
        mapLowRes = predict(detector, single(img_resized));
        detections = zeros(3, 0);
        detMasks = {};

        for k = 1:numAnomalyClasses
            mapLowResClass = mapLowRes(:,:,k);

            % 1. Get current frame score
            meanScore = mean(mapLowResClass, 'all');

            % 4. Gatekeeper: Check the smoothed score against the threshold
            if meanScore < thresholds(k)
                continue;
            end

            % 5. Blob Extraction: Proceed only if the gatekeeper passes
            mapResized = imresize(mapLowResClass, [imgH imgW], "bilinear");
            binMask = mapResized >= thresholds(k);

            if ~any(binMask(:)), continue; end

            cc = bwconncomp(binMask);
            props = regionprops(cc, 'Centroid', 'Area', 'PixelIdxList');

            for p = 1:length(props)
                if props(p).Area < cfg.minArea, continue; end

                detections = [detections, [props(p).Centroid(1); props(p).Centroid(2); k]];
                blobMask = false(imgH, imgW);
                blobMask(props(p).PixelIdxList) = true;
                detMasks{end+1} = blobMask;
            end
        end
    end

    % 2. EKF Logic
    [matchings, unassigned_dets, unassigned_tracks] = perform_gnn_association(tracks, detections, xr_tether, estimated_pp, cfg);
    tracks = update_existing_tracks(tracks, detections, matchings, unassigned_tracks, xr_tether, estimated_pp, cfg);
    [tracks, state] = initialize_new_tracks(tracks, detections, unassigned_dets, xr_tether, estimated_pp, state, cfg);
    
    % 3. Mask Rendering 
    persistentAnomalyMask(:) = 0; 
    if ~isempty(matchings)
        for m = 1:size(matchings, 1)
            dIdx = matchings(m,1); tIdx = matchings(m,2);
            if tracks(tIdx).hits >= cfg.min_hits
                m_color = tracks(tIdx).color;
                for c = 1:3
                    % Blend detected mask into the color channels
                    persistentAnomalyMask(:,:,c) = persistentAnomalyMask(:,:,c) + double(detMasks{dIdx}) * m_color(c);
                end
            end
        end
    end
    
    maskWeight = 0.5;
    hasMask = any(persistentAnomalyMask, 3) > 0;
    if any(hasMask(:))
        for c = 1:3
            imgChannel = double(img(:,:,c));
            % Slice the 3D mask correctly for the current channel
            maskChannel = persistentAnomalyMask(:,:,c);
            % Apply alpha blending to the pixels in the mask
            imgChannel(hasMask) = (1-maskWeight) * imgChannel(hasMask) + maskWeight * (maskChannel(hasMask)*255);
            img(:,:,c) = uint8(imgChannel);
        end
    end

    % Merge tracks AFTER we are done using the 'matchings' indices ---
    tracks = merge_redundant_tracks(tracks, cfg);


   % 4. Static 3D Wiremesh Overlay Calculation
    % Define the physical boundaries of your wireframe (in millimeters)
    mesh_start_Z = cfg.mesh_start_Z;  % How close to the lens the mesh starts (150mm)
    mesh_end_Z = cfg.mesh_end_Z;  % How far down the tunnel the mesh goes (5 meters)
    ring_spacing = 100;  % Put a ring every 100mm
    
    % 1. Calculate the Concentric Rings
    ringDists = mesh_start_Z : ring_spacing : mesh_end_Z;
    ringPtsX = []; ringPtsY = [];
    
    for d = ringDists
        theta_vec = linspace(0, 2*pi, 40);
        rho = (cfg.f_px * cfg.pipe_R) / d;
        ringPtsX = [ringPtsX, estimated_pp(1) + rho*cos(theta_vec), nan];
        ringPtsY = [ringPtsY, estimated_pp(2) + rho*sin(theta_vec), nan];
    end
    
    % 2. Calculate the Longitudinal Depth Lines
    num_depth_lines = 12; % Number of lines running down the pipe
    angles = linspace(0, 2*pi, num_depth_lines + 1);
    angles(end) = []; % Remove duplicate 2*pi at the end
    
    for t = angles
        % Find the pixel radius at the furthest point
        rho_far = (cfg.f_px * cfg.pipe_R) / mesh_end_Z;
        u_far = estimated_pp(1) + rho_far * cos(t);
        v_far = estimated_pp(2) + rho_far * sin(t);
        
        % Find the pixel radius at the closest point
        rho_near = (cfg.f_px * cfg.pipe_R) / mesh_start_Z;
        u_near = estimated_pp(1) + rho_near * cos(t);
        v_near = estimated_pp(2) + rho_near * sin(t);
        
        % Draw a line connecting them
        ringPtsX = [ringPtsX, u_near, u_far, nan];
        ringPtsY = [ringPtsY, v_near, v_far, nan];
    end
    
    % Update the single plot object with all lines at once
    set(hMeshRings, 'XData', ringPtsX, 'YData', ringPtsY);

    % 5. 3D Plotting and Metric Axis Re-labeling
    set(hVideoImg, 'CData', img);
    set(hSpiderCenter, 'XData', estimated_pp(1), 'YData', estimated_pp(2));
    
    % Animate the pipe wireframe sliding backwards
    meshShift = mod(xr_tether, ringSpacing);
    set(hPipeMesh, 'XData', X_grid - meshShift);
    
    T = makehgtform('translate', [0, 0, -cfg.pipe_R/2], 'zrotate', pi, 'scale', 400);
    set(hRoverGroup, 'Matrix', T);

    relativeTicks = -1000:1000:4000; 
    absoluteMeters = (xr_tether + relativeTicks) / 1000;
    tickLabels = arrayfun(@(m) sprintf('%.1f', m), absoluteMeters, 'UniformOutput', false);
    set(axMap3D, 'XTick', relativeTicks, 'XTickLabel', tickLabels);

    numTracks = length(tracks);

    % Dynamically expand the graphics pools if tracks exceed current capacity
    if numTracks > length(hVideoBBoxes)
        oldSize = length(hVideoBBoxes);
        for newIdx = (oldSize + 1):numTracks
            hDetPool(newIdx)     = plot(axVideo, nan, nan, 's', 'MarkerSize', 10, 'LineWidth', 2, 'Visible', 'off');
            hMapPoints3D(newIdx) = scatter3(axMap3D, nan, nan, nan, 100, 'filled', 'MarkerEdgeColor', 'k', 'Visible', 'off');
            hMapLabels3D(newIdx) = text(axMap3D, 0, 0, 0, '', 'FontSize', 9, 'FontWeight', 'bold', 'Visible', 'off');
            hVideoBBoxes(newIdx) = rectangle(axVideo, 'Position', [0 0 1 1], 'EdgeColor', 'w', 'LineWidth', 2, 'Visible', 'off');
            hVideoLabels(newIdx) = text(axVideo, 0, 0, '', 'FontSize', 10, 'FontWeight', 'bold', 'Color', 'w', 'Visible', 'off', 'Clipping', 'on');
        end
    end

    for i = 1:numTracks
        if tracks(i).hits >= cfg.min_hits
            relX = tracks(i).mu(1) - xr_tether;
            
            % --- 3D MAP PLOTTING ---
            theta_adj = tracks(i).mu(2) + cfg.angle_offset;
            draw_R = cfg.pipe_R * 0.95;
            worldY = draw_R * cos(theta_adj);
            worldZ = draw_R * sin(theta_adj);
            
            if relX > -1000 && relX < 4000
                set(hMapPoints3D(i), 'XData', relX, 'YData', worldY, 'ZData', worldZ, ...
                    'CData', tracks(i).color, 'Visible', 'on');
                set(hMapLabels3D(i), 'Position', [relX, worldY, worldZ + 60], ...
                    'String', sprintf('ID:%d', tracks(i).id), 'Color', tracks(i).color, 'Visible', 'on');
            else
                set(hMapPoints3D(i), 'Visible', 'off'); set(hMapLabels3D(i), 'Visible', 'off');
            end
            
            % --- 2D VIDEO PROJECTION ---
            if relX > 50 % Only draw if the anomaly is physically in front of the camera lens
                % Project the stable EKF state back into 2D pixel coordinates
                rho = (cfg.f_px * cfg.pipe_R) / relX;
                u = estimated_pp(1) + rho * cos(tracks(i).mu(2));
                v = estimated_pp(2) + rho * sin(tracks(i).mu(2));
                
                % Scale the bounding box size dynamically based on distance
                % (Assuming a fixed physical size of ~200mm on the pipe wall)
                boxSize = (cfg.f_px * 200) / relX; 
                
                % Update the video bounding box and label
                set(hVideoBBoxes(i), 'Position', [u - boxSize/2, v - boxSize/2, boxSize, boxSize], ...
                    'EdgeColor', tracks(i).color, 'Visible', 'on');
                set(hVideoLabels(i), 'Position', [u - boxSize/2, v - boxSize/2 - 10], ...
                    'String', sprintf('ID:%d', tracks(i).id), 'Color', tracks(i).color, 'Visible', 'on');
            else
                % Turn off if the robot drove past it
                set(hVideoBBoxes(i), 'Visible', 'off'); set(hVideoLabels(i), 'Visible', 'off');
            end
            
        else
            % Turn everything off if the track doesn't exist or isn't mature
            set(hMapPoints3D(i), 'Visible', 'off'); set(hMapLabels3D(i), 'Visible', 'off');
            set(hVideoBBoxes(i), 'Visible', 'off'); set(hVideoLabels(i), 'Visible', 'off');
        end
    end
    
    drawnow limitrate;
    writeVideo(vWriter, getframe(hFig));
    
    % ==========================================================
    % CAPTURE VECTOR GRAPHICS FOR THE FINAL FIGURE
    % ==========================================================
    if frameCount == targetFrameNum
        disp(['Cloning vector graphics at frame ', num2str(frameCount), '...']);
        
        % Create an invisible "vault" figure so it doesn't pop up and interrupt you
        hVaultFig = figure('Visible', 'off'); 
        
        % Clone the entire video axis (and all its lines/boxes) into the vault
        axVaultVideo = copyobj(axVideo, hVaultFig); 
    end

end
close(vWriter);

% =========================================================================
% POST-PROCESSING: SPATIAL HEURISTIC FILTER (Remove Ceiling Deposits)
% =========================================================================
disp('Applying Post-Processing Spatial Filter to Final Map...');

% Loop backwards to safely delete elements from the array
for i = length(tracks):-1:1
    
    % Check if the track is a Deposit (Class 2)
    if tracks(i).class == 2 
        
        % Calculate its true 3D vertical position (Z-axis)
        theta_adj = tracks(i).mu(2) + cfg.angle_offset;
        worldZ = cfg.pipe_R * sin(theta_adj);
        
        % If worldZ is greater than 0, it is in the top half of the pipe
        if worldZ > 0
            tracks(i) = []; % Erase the false positive track completely
        end
        
    end
end

% =========================================================================
% POST-PROCESSING: SPATIAL HEURISTIC FILTER (Remove Bottom Connections)
% =========================================================================
disp('Applying Post-Processing Spatial Filter to Final Map...');

% Loop backwards to safely delete elements from the array
for i = length(tracks):-1:1
    
    % Check if the track is a Deposit (Class 2)
    if tracks(i).class == 1 
        
        % Calculate its true 3D vertical position (Z-axis)
        theta_adj = tracks(i).mu(2) + cfg.angle_offset;
        worldZ = cfg.pipe_R * sin(theta_adj);
        
        % If worldZ is less than 0, it is in the bottom half of the pipe
        if worldZ < 0
            tracks(i) = []; % Erase the false positive track completely
        end
        
    end
end


%% 
% =========================================================================
% FINAL STATIC FIGURE: IEEE STANDARD (3-PANEL WITH LEGEND)
% =========================================================================
% --- Configuration for Final Figure ---
targetImg = read(vReader, targetFrameNum);
xr_target = pipeOdom(targetFrameNum); 

% Define absolute bounds based on analyzed frames
minPipeDist = pipeOdom(startFrame);
maxPipeDist = pipeOdom(endFrame);

% --- 1. Create the IEEE-Formatted Figure ---
hFigFinal = figure('Color', 'w', 'Units', 'centimeters', 'Position', [2 2 18 6], ...
                   'Name', 'Final Global Perspective');

% 2x5 layout: 2 rows, 5 columns total.
t = tiledlayout(2, 6, 'TileSpacing', 'tight', 'Padding', 'tight');

% -------------------------------------------------------------------------
% TOP-LEFT PANEL: Original CCTV Image (Spans 1 row, 2 columns)
% -------------------------------------------------------------------------
axTopLeft = nexttile(1, [1 2]); 
hold(axTopLeft, 'on');
imshow(targetImg, 'Parent', axTopLeft);
xlim(axTopLeft, [1 imgW]); 
ylim(axTopLeft, [1 imgH]);
title(axTopLeft, 'Original CCTV Frame', ...
    'FontName', 'Times New Roman', 'FontSize', 9, 'FontWeight', 'normal');

% -------------------------------------------------------------------------
% BOTTOM-LEFT PANEL: Static Video Snapshot (Spans 1 row, 2 columns)
% -------------------------------------------------------------------------
axBotLeft = nexttile(7, [1 2]); 
hold(axBotLeft, 'on');

% Load the perfectly matched snapshot from your live video loop
% (Ensure this filename matches what you saved!)
snapshotImg = imread('VideoSnapshot.png'); 
imshow(snapshotImg, 'Parent', axBotLeft);

title(axBotLeft, 'Anomaly Overlay & Tracking', ...
    'FontName', 'Times New Roman', 'FontSize', 9, 'FontWeight', 'normal');
axis(axBotLeft, 'off'); % Ensures no stray borders appear around your saved image

% -------------------------------------------------------------------------
% RIGHT PANEL: Global 3D Cylindrical Map (Spans 2 rows, 3 columns)
% -------------------------------------------------------------------------
axRight = nexttile(3, [2 4]); 
hold(axRight, 'on'); grid(axRight, 'on');
title(axRight, '3D Pipe Reconstruction', 'FontName', 'Times New Roman', 'FontSize', 9, 'FontWeight', 'normal');
view(axRight, [-35, 25]); 
xlim(axRight, [minPipeDist, maxPipeDist]);
axis(axRight, 'tight'); 

% Compress the longitudinal axis so the pipe has thickness
lengthCompression = 5; 
daspect(axRight, [lengthCompression, 1, 1]); 
zlabel(axRight, 'Z (mm)', 'FontName', 'Times New Roman', 'FontSize', 8); 
ylabel(axRight, 'Y (mm)', 'FontName', 'Times New Roman', 'FontSize', 8); 
xlabel(axRight, 'Distance (m)', 'FontName', 'Times New Roman', 'FontSize', 8);
set(axRight, 'FontName', 'Times New Roman', 'FontSize', 8, 'LineWidth', 0.5);

% Dynamically calculate mesh step size
meshStep = (maxPipeDist - minPipeDist) / 50; 
[ThetaFull, X_gridFull] = meshgrid(linspace(0, 2*pi, 40), minPipeDist:meshStep:maxPipeDist);
cylYFull = cfg.pipe_R * cos(ThetaFull);
cylZFull = cfg.pipe_R * sin(ThetaFull);
mesh(axRight, X_gridFull, cylYFull, cylZFull, 'FaceAlpha', 0.05, 'EdgeColor', [0.7 0.7 0.8], 'LineWidth', 0.5);
xTickVals = linspace(minPipeDist, maxPipeDist, 5);
set(axRight, 'XTick', xTickVals, 'XTickLabel', round(xTickVals/1000, 1));
yZTickVals = [-cfg.pipe_R, 0, cfg.pipe_R];
set(axRight, 'YTick', yZTickVals, 'ZTick', yZTickVals);

% --- Legend Tracking Setup ---
ekfHandles = []; ekfLabels = {};
gtHandles = [];  gtLabels = {};
addedDefectTypes = {}; % Keep track of what we've already added

% ---------------------------------------------------------
% 1. PLOT EKF TRACKS (Filled Circles)
% ---------------------------------------------------------
for i = 1:length(tracks)
    if tracks(i).hits >= cfg.min_hits
        globalX = tracks(i).mu(1); 
        
        if globalX >= minPipeDist && globalX <= maxPipeDist
            theta_adj = tracks(i).mu(2) + cfg.angle_offset;
            draw_R = cfg.pipe_R * 0.95; 
            worldY = draw_R * cos(theta_adj);
            worldZ = draw_R * sin(theta_adj);
            
            % Plot Tracked Anomaly as a circle
            hDot = scatter3(axRight, globalX, worldY, worldZ, 25, 'filled', ...
                'MarkerEdgeColor', 'k', 'MarkerFaceColor', tracks(i).color, 'LineWidth', 0.5);
            
            % Setup label (e.g., "Deposit")
            classIdx = tracks(i).class; 
            defectType = sprintf('%s', classNames(classIdx)); 
            
            if ~ismember(defectType, addedDefectTypes)
                ekfHandles(end+1) = hDot;
                ekfLabels{end+1} = defectType;
                addedDefectTypes{end+1} = defectType; 
            end
        end
    end
end

% ---------------------------------------------------------
% 2. PLOT GROUND TRUTH DATA (Large Filled Diamonds)
% ---------------------------------------------------------
validGT_idx = (pipeOdom >= minPipeDist) & (pipeOdom <= maxPipeDist);

% --- GT Connections (Class 1) - Plotted at the TOP of the pipe ---
idx_conn = (gTruth.Connection == 1) & validGT_idx;
if any(idx_conn)
    x_c = pipeOdom(idx_conn);
    y_c = zeros(size(x_c));
    z_c = repmat(cfg.pipe_R, size(x_c)); 
    
    % Increased size to 120, made it filled, with a distinct black outline
    hGTConn = scatter3(axRight, x_c, y_c, z_c, 30, 'd', 'filled', ...
        'MarkerFaceColor', cfg.baseColors(1,:), ...
        'MarkerEdgeColor', 'k', ...
        'LineWidth', 0.8);
    
    gtHandles(end+1) = hGTConn;
    gtLabels{end+1} = sprintf('%s', classNames(1));
end

% --- GT Deposits (Class 2) - Plotted at the BOTTOM of the pipe ---
idx_dep = (gTruth.Deposit == 1) & validGT_idx;
if any(idx_dep)
    x_d = pipeOdom(idx_dep);
    y_d = zeros(size(x_d));
    z_d = repmat(-cfg.pipe_R, size(x_d)); 
    
    % Increased size to 120, made it filled, with a distinct black outline
    hGTDep = scatter3(axRight, x_d, y_d, z_d, 50, 'd', 'filled', ...
        'MarkerFaceColor', cfg.baseColors(2,:), ...
        'MarkerEdgeColor', 'k', ...
        'LineWidth', 0.8);
    
    gtHandles(end+1) = hGTDep;
    gtLabels{end+1} = sprintf('%s', classNames(2));
end

% ---------------------------------------------------------
% 3. DRAW DUAL LEGENDS
% ---------------------------------------------------------
% --- Legend 1: EKF Detections ---
lgd1 = legend(axRight, ekfHandles, ekfLabels, 'Location', 'none', ...
    'FontName', 'Times New Roman', 'FontSize', 8, 'AutoUpdate', 'off');
title(lgd1, 'Detections', 'FontName', 'Times New Roman', 'FontSize', 8);
lgd1.Color = 'w';
lgd1.EdgeColor = [0.8 0.8 0.8];
lgd1.Position = [0.265 0.67 0.15 0.32]; 

% --- Create Invisible Dummy Axes ---
axDummy = axes('Parent', hFigFinal, 'Position', axRight.Position, ...
               'Color', 'none', 'XColor', 'none', 'YColor', 'none', 'HitTest', 'off');

% --- Legend 2: Ground Truth ---
lgd2 = legend(axDummy, gtHandles, gtLabels, 'Location', 'none', ...
    'FontName', 'Times New Roman', 'FontSize', 8, 'AutoUpdate', 'off');
title(lgd2, 'Ground Truth', 'FontName', 'Times New Roman', 'FontSize', 8);
lgd2.Color = 'w';
lgd2.EdgeColor = [0.8 0.8 0.8];
lgd2.Position = [0.265 0.475 0.15 0.2]; 

% Save the map
exportgraphics(hFigFinal, 'Final_Mapping.png', 'Resolution', 600);
