function outputImage = resizeImage(inputImage, targetSize)
   
    % Get original size
    [origH, origW, ~] = size(inputImage);

    % Compute scaling factor while preserving aspect ratio
    scale = min(targetSize(1)/origH, targetSize(2)/origW);

    % Resize the image with the scale
    newH = round(origH * scale);
    newW = round(origW * scale);
    resizedImage = imresize(inputImage, [newH, newW]);

    % Create a black canvas of 224x224
    outputImage = zeros([targetSize, 3], 'like', inputImage);

    % Compute padding offsets
    top = floor((targetSize(1) - newH) / 2) + 1;
    left = floor((targetSize(2) - newW) / 2) + 1;

    % Place resized image onto canvas
    outputImage(top:top+newH-1, left:left+newW-1, :) = resizedImage;
    
end