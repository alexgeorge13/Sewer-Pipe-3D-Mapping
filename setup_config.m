function cfg = setup_config()
    cfg.f_px = 200; % Updated from calibration!
    cfg.pipe_R = 300; cfg.R_sensor = diag([15, 15].^2);
    cfg.gate_limit = 9.21; cfg.cost_unmatched = 9.21;
    cfg.baseColors = [0.02 1.0 0.65; 1.0 0.1 0.6; 0.15 0.6 0.85; 0.9 0.7 0.1; 0.4 0.1 0.9];
    cfg.minArea = 3000; cfg.pp_window = 100;
    cfg.angle_offset = pi; 
    cfg.min_dist = 1000;
    cfg.min_hits = 5; % Change this number to require more or fewer frames!

    % --- NEW: 3D Tracking Region of Interest ---
    cfg.mesh_start_Z = 0;  % Minimum trackable distance (mm)
    cfg.mesh_end_Z = 500;   % Maximum trackable distance (mm)
end