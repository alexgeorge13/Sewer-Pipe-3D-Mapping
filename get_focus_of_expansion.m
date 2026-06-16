function [u, v, inliers, outliers, X_val, Y_val, Vx_val, Vy_val] = get_focus_of_expansion(flow, imgW, imgH)
    Vx = flow.Vx(:); Vy = flow.Vy(:);
    [X, Y] = meshgrid(1:size(flow.Vx,2), 1:size(flow.Vx,1));
    X = X(:); Y = Y(:);
    
    mag = Vx.^2 + Vy.^2;
    validFocus = mag > prctile(mag, 20); 
    
    Vx_val = Vx(validFocus); Vy_val = Vy(validFocus); 
    X_val = X(validFocus); Y_val = Y(validFocus);
    
    A = [-Vy_val, Vx_val]; 
    b = -Vy_val .* X_val + Vx_val .* Y_val;
    norms = sqrt(mag(validFocus)); 
    
    % --- SPEED FIX: Spatial Decimation ---
    maxVectors = 400; % Increased capacity since LK is so fast
    if length(b) > maxVectors
        subIdx = round(linspace(1, length(b), maxVectors)); 
        A = A(subIdx, :); b = b(subIdx); norms = norms(subIdx);
        X_val = X_val(subIdx); Y_val = Y_val(subIdx); 
        Vx_val = Vx_val(subIdx); Vy_val = Vy_val(subIdx);
    end
    
    numVectors = size(A, 1);
    if numVectors < 2
        u = imgW/2; v = imgH/2; inliers = []; outliers = []; return;
    end
    
    % --- UPGRADE: Adaptive Early Stopping RANSAC ---
    maxIterations = 1000;     
    currentMaxIter = maxIterations;
    inlierThreshold = 3.0;    
    bestInlierCount = 0;
    bestModelInliers = [];
    desiredConfidence = 0.99; % 99% confidence requirement
    
    for i = 1:maxIterations
        idx = randperm(numVectors, 2);
        A_rand = A(idx, :); b_rand = b(idx);
        if abs(det(A_rand)) < 1e-5, continue; end
        
        candidate_center = A_rand \ b_rand;
        distances = abs(A * candidate_center - b) ./ norms;
        currentInliers = find(distances < inlierThreshold);
        currentCount = length(currentInliers);
        
        if currentCount > bestInlierCount
            bestInlierCount = currentCount;
            bestModelInliers = currentInliers;
            
            % Dynamically shrink max iterations if a great model is found
            w = bestInlierCount / numVectors; 
            if w > 0
                N = log(1 - desiredConfidence) / log(1 - w^2);
                currentMaxIter = min(maxIterations, ceil(N)); 
            end
        end
        
        % Quit immediately if we hit the mathematical confidence threshold
        if i >= currentMaxIter
            break; 
        end
    end
    
    inliers = bestModelInliers;
    outliers = setdiff(1:numVectors, inliers);
    
    if bestInlierCount > 5
        final_center = A(inliers, :) \ b(inliers);
    else
        final_center = A \ b; 
    end
    
    u = min(max(final_center(1) * (imgW / size(flow.Vx,2)), 0), imgW); 
    v = min(max(final_center(2) * (imgH / size(flow.Vx,1)), 0), imgH);
end