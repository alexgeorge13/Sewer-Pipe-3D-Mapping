function tracks = update_existing_tracks(tracks, detections, matchings, unassigned_tracks, xr, pp, cfg)

    % 1. Increment misses for tracks that didn't get a detection
    for i = 1:length(unassigned_tracks)
        trk_idx = unassigned_tracks(i);
        tracks(trk_idx).misses = tracks(trk_idx).misses + 1;
    end

    if isempty(matchings), return; end

    for k = 1:size(matchings, 1)
        det_idx = matchings(k, 1); trk_idx = matchings(k, 2);
        z = detections(1:2, det_idx);
        tracks(trk_idx).hits = tracks(trk_idx).hits + 1;
        tracks(trk_idx).misses = 0; % Reset misses if it was found again

        lx = tracks(trk_idx).mu(1); lt = tracks(trk_idx).mu(2);
        dx_est = lx - xr;
        if dx_est > 50
            rho_p = (cfg.f_px * cfg.pipe_R) / dx_est;
            z_p = [pp(1) + rho_p * cos(lt); pp(2) + rho_p * sin(lt)];
            H = [ -(cfg.f_px*cfg.pipe_R*cos(lt))/(dx_est^2), -(cfg.f_px*cfg.pipe_R*sin(lt))/dx_est; ...
                -(cfg.f_px*cfg.pipe_R*sin(lt))/(dx_est^2),  (cfg.f_px*cfg.pipe_R*cos(lt))/dx_est ];
            S = H * tracks(trk_idx).P * H' + cfg.R_sensor;
            K = tracks(trk_idx).P * H' / S;
            tracks(trk_idx).mu = tracks(trk_idx).mu + K * (z - z_p);
            tracks(trk_idx).P = (eye(2) - K * H) * tracks(trk_idx).P;
        end
    end
end