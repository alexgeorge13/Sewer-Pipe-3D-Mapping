function tracks = merge_redundant_tracks(tracks, cfg)
    % If there's 1 or 0 tracks, there's nothing to merge
    if length(tracks) < 2
        return;
    end

    % Keep a logical array of tracks to delete so we don't mess up loop indexing
    to_delete = false(length(tracks), 1);

    for i = 1:length(tracks)
        if to_delete(i)
            continue; % Skip if this track was already absorbed
        end

        % Convert Track A's state (odometry, theta) to 3D Cartesian coordinates
        x1 = tracks(i).mu(1);
        y1 = cfg.pipe_R * cos(tracks(i).mu(2));
        z1 = cfg.pipe_R * sin(tracks(i).mu(2));

        for j = (i+1):length(tracks)
            % Skip if Track B is already marked for deletion or if they are different classes
            if to_delete(j) || any(tracks(i).class ~= tracks(j).class)
                continue;
            end

            % Convert Track B's state to 3D Cartesian coordinates
            x2 = tracks(j).mu(1);
            y2 = cfg.pipe_R * cos(tracks(j).mu(2));
            z2 = cfg.pipe_R * sin(tracks(j).mu(2));

            % Calculate the true 3D Euclidean distance between the two tracks
            dist = sqrt((x1 - x2)^2 + (y1 - y2)^2 + (z1 - z2)^2);

            % If they are closer than the threshold, merge them!
            if dist < cfg.min_dist
                % Keep the more mature track (highest hits)
                if tracks(i).hits >= tracks(j).hits
                    % Track A absorbs Track B
                    tracks(i).hits = tracks(i).hits + tracks(j).hits; 
                    to_delete(j) = true;
                else
                    % Track B absorbs Track A
                    tracks(j).hits = tracks(j).hits + tracks(i).hits;
                    to_delete(i) = true;
                    break; % Track i is dead, stop comparing it to others
                end
            end
        end
    end

    % Strip all the deleted tracks out of the struct array in one go
    tracks(to_delete) = [];
end