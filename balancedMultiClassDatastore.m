classdef balancedMultiClassDatastore < matlab.io.Datastore & ...
        matlab.io.datastore.Partitionable & ...
        matlab.io.datastore.Shuffleable

    properties
        classDS
        classNames
        numClasses
        seenFlags
    end

    methods
        function self = balancedMultiClassDatastore(dsArray, names)
            self.classDS = cellfun(@copy, dsArray, 'UniformOutput', false);
            self.classNames = names;
            self.numClasses = numel(dsArray);

            % Track coverage (robust to imds / arrayDatastore)
            classCounts = zeros(1, self.numClasses);
            for k = 1:self.numClasses
                d = dsArray{k};
                if isprop(d, 'Files')
                    classCounts(k) = numel(d.Files);
                elseif isa(d, 'matlab.io.datastore.ArrayDatastore')
                    classCounts(k) = numel(d);   % works for arrayDatastore

                else
                    error('Unsupported datastore type for class %d', k);
                end
            end

            self.seenFlags = cellfun(@(n) false(n,1), ...
                num2cell(classCounts), 'UniformOutput', false);
        end

        function [dataOut,info] = read(self)
            if ~hasdata(self)
                error('No more data. Call reset to start next epoch.');
            end

            % Pick a class uniformly
            k = randi(self.numClasses);

            if ~hasdata(self.classDS{k})
                reset(self.classDS{k});
                self.classDS{k} = shuffle(self.classDS{k});
            end

            [X,info] = read(self.classDS{k});
            data = iUnpackData(X);

            % Mark image as seen using our own counter
            if isfield(info,'Filename') % imageDatastore
                idx = find(strcmp(self.classDS{k}.Files, info.Filename), 1);
                if ~isempty(idx)
                    self.seenFlags{k}(idx) = true;
                end
            elseif isfield(info,'Index') % arrayDatastore
                self.seenFlags{k}(info.Index) = true;
            end


            label = self.classNames(k);
            dataOut = {data, label};
        end

        function TF = hasdata(self)
            TF = ~all(cellfun(@all, self.seenFlags));
        end

        function reset(self)
            for k = 1:self.numClasses
                reset(self.classDS{k});
                self.classDS{k} = shuffle(self.classDS{k});
                self.seenFlags{k}(:) = false;
            end
        end

        % ===== Required abstract methods =====
        function dsNew = shuffle(self)
            dsArray = cell(1,self.numClasses);
            for k = 1:self.numClasses
                dsArray{k} = shuffle(self.classDS{k});
            end
            dsNew = balancedMultiClassDatastore(dsArray, self.classNames);
        end

        function dsNew = partition(self, N, ii)
            dsArray = cell(1,self.numClasses);
            for k = 1:self.numClasses
                dsArray{k} = partition(self.classDS{k},N,ii);
            end
            dsNew = balancedMultiClassDatastore(dsArray, self.classNames);
        end
    end

    methods (Access = protected)
        function n = maxpartitions(self)
            n = min(cellfun(@numpartitions, self.classDS));
        end
    end
end

function data = iUnpackData(data)
if ~isnumeric(data)
    data = data{1};
end
end