function [tracks, state] = initialize_new_tracks(tracks, detections, unassigned, xr, pp, state, cfg)
    if isempty(unassigned) || isempty(detections), return; end
    
    % The (:)' guarantees it is ALWAYS a row vector so it iterates one-by-one!
    for d_idx = unassigned(:)'
        z = detections(1:2, d_idx); det_class = detections(3, d_idx);
        du = z(1) - pp(1); dv = z(2) - pp(2);
        est_rho = sqrt(du^2 + dv^2);
        
        if est_rho > 1 
            est_dx = (cfg.f_px * cfg.pipe_R) / est_rho;
            
            % --- NEW: THE 3D DEPTH GATE ---
            % If the calculated distance is closer than the start of the mesh
            % OR further than the end of the mesh, ignore it completely!
            if est_dx < cfg.mesh_start_Z || est_dx > cfg.mesh_end_Z
                continue; % Skip this detection and move to the next one
            end
            
            % If it passed the gate, initialize the track!
            est_theta = atan2(dv, du);
            new_trk.mu = [xr + est_dx; est_theta];
            new_trk.P = diag([500^2, 0.3^2]);
            new_trk.id = state.next_id;
            new_trk.class = det_class;
            new_trk.hits = 1; new_trk.misses = 0;
            new_trk.color = cfg.baseColors(det_class, :);
            if isempty(tracks), tracks = new_trk; else, tracks(end+1) = new_trk; end
            state.next_id = state.next_id + 1;
        end
    end
end