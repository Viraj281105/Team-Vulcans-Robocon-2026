classdef ImageLoader
    % IMAGELOADER Utility class for loading oracle bone images
    
    methods (Static)
        function [realImgs, fakeImgs] = loadOracleBones(realPath, fakePath)
            % Load oracle bone images from specified paths
            realImgs = {};
            fakeImgs = {};
            
            if ~isempty(realPath) && exist(realPath, 'dir')
                files = [dir(fullfile(realPath, '*.png')); ...
                        dir(fullfile(realPath, '*.jpg'))];
                        
                for f = 1:numel(files)
                    try
                        img = imread(fullfile(realPath, files(f).name));
                        if size(img, 3) == 1
                            img = repmat(img, [1 1 3]);
                        end
                        realImgs{end+1} = img; %#ok<AGROW>
                    catch
                        fprintf('[WARN] Failed to load: %s\n', files(f).name);
                    end
                end
            end
            
            if ~isempty(fakePath) && exist(fakePath, 'dir')
                files = [dir(fullfile(fakePath, '*.png')); ...
                        dir(fullfile(fakePath, '*.jpg'))];
                        
                for f = 1:numel(files)
                    try
                        img = imread(fullfile(fakePath, files(f).name));
                        if size(img, 3) == 1
                            img = repmat(img, [1 1 3]);
                        end
                        fakeImgs{end+1} = img; %#ok<AGROW>
                    catch
                        fprintf('[WARN] Failed to load: %s\n', files(f).name);
                    end
                end
            end
            
            fprintf('[OK] Loaded %d REAL, %d FAKE oracle bone images\n', ...
                    numel(realImgs), numel(fakeImgs));
        end
    end
end
