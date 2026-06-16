function [matchings, unassigned_dets, unassigned_tracks] = perform_gnn_association(tracks, detections, xr, pp, cfg)
    num_detections = size(detections, 2);
    num_tracks = length(tracks);
    matchings = []; unassigned_dets = 1:num_detections; unassigned_tracks = 1:num_tracks;
    if num_detections == 0 || num_tracks == 0, return; end
    Cost = inf(num_detections, num_tracks);
    for d = 1:num_detections
        z = detections(1:2, d);
        det_class = detections(3, d);
        for i = 1:num_tracks
            if isfield(tracks, 'class') && any(tracks(i).class ~= det_class), continue; end
            dx_est = tracks(i).mu(1) - xr;
            if dx_est > 50
                lt = tracks(i).mu(2);
                rho_p = (cfg.f_px * cfg.pipe_R) / dx_est;
                z_p = [pp(1) + rho_p * cos(lt); pp(2) + rho_p * sin(lt)];
                H = [ -(cfg.f_px*cfg.pipe_R*cos(lt))/(dx_est^2), -(cfg.f_px*cfg.pipe_R*sin(lt))/dx_est; ...
                      -(cfg.f_px*cfg.pipe_R*sin(lt))/(dx_est^2),  (cfg.f_px*cfg.pipe_R*cos(lt))/dx_est ];
                S = H * tracks(i).P * H' + cfg.R_sensor;
                y = z - z_p;
                dm = y' * (S \ y); 
                if dm < cfg.gate_limit, Cost(d, i) = dm; end
            end
        end
    end
    [matchings, unassigned_dets] = matchpairs(Cost, cfg.cost_unmatched);

    % NEW: Find tracks that are not in the matchings list
    if ~isempty(matchings)
        unassigned_tracks = setdiff(1:num_tracks, matchings(:, 2)');
    end

end