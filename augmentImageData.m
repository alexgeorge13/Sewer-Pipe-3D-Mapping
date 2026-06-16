function [dataOut, info] = augmentImageData(dataIn, info, targetSize, params)

if ~iscell(dataIn)
    dataIn = {dataIn};
end

dataOut = cell([size(dataIn,1), 2]);

for idx = 1:size(dataIn,1)
    data = dataIn{idx};

    % Convert grayscale → RGB if needed
    if size(data, 3) == 1
        data = repmat(data, [1 1 3]);
    end

    % Resize
    data = resizeImage(data, targetSize);

    % ----------------------------
    % Decide whether to augment
    % ----------------------------
    if rand < params.AugProb
        % Build a list of enabled augmentations
        augList = {};

        if ~isempty(params.RandRotation),      augList{end+1} = 'rotation'; end
        if ~isempty(params.RandXTranslation) || ~isempty(params.RandYTranslation)
            augList{end+1} = 'translation'; 
        end
        if params.RandXReflection || params.RandYReflection, augList{end+1} = 'reflection'; end
        if ~isempty(params.RandBrightness),    augList{end+1} = 'brightness'; end
        if params.RandContrast,                augList{end+1} = 'contrast'; end
        if params.RandColorJitter,             augList{end+1} = 'colorjitter'; end
        if ~isempty(params.RandGaussianBlur),  augList{end+1} = 'gaussianblur'; end
        if params.RandNoise,                   augList{end+1} = 'noise'; end

        % Randomly pick one augmentation
        if ~isempty(augList)
            choice = augList{randi(numel(augList))};

            switch choice
                case 'rotation'
                    angle = randi(params.RandRotation);
                    data = imrotate(data, angle, 'crop');

                case 'translation'
                    tx = 0; ty = 0;
                    if ~isempty(params.RandXTranslation)
                        tx = randi(params.RandXTranslation);
                    end
                    if ~isempty(params.RandYTranslation)
                        ty = randi(params.RandYTranslation);
                    end
                    data = imtranslate(data, [tx, ty], 'FillValues', 0);

                case 'reflection'
                    if params.RandXReflection && rand >= 0.5
                        data = fliplr(data);
                    end
                    if params.RandYReflection && rand >= 0.5
                        data = flipud(data);
                    end

                case 'brightness'
                    factor = params.RandBrightness(1) + ...
                             (params.RandBrightness(2)-params.RandBrightness(1))*rand();
                    data = imadjust(data, [], [], factor);

                case 'contrast'
                    data = imadjust(data, stretchlim(data), []);

                case 'colorjitter'
                    hueShift = rand()*0.4;
                    data = jitterColorHSV(data, ...
                        'Hue',[-hueShift, hueShift], ...
                        'Saturation',[-0.1, 0.1], ...
                        'Brightness',[-0.1, 0.1]);

                case 'gaussianblur'
                    sigma = params.RandGaussianBlur(1) + ...
                            rand()*(params.RandGaussianBlur(2)-params.RandGaussianBlur(1));
                    data = imgaussfilt(data, sigma);

                case 'noise'
                    data = imnoise(data,"gaussian");
            end
        end
    end

    % Output
    dataOut(idx,:) = {data, info.Label};
end
end
