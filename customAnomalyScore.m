function Y = customAnomalyScore(X)
    % This replaces the anonymous function so the quantizer can read it!
    Y = sqrt(X.^2+1)-1; 
end