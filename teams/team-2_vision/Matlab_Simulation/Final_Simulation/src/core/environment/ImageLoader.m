classdef ImageLoader
    % ===================================================================
    % IMAGELOADER
    % -------------------------------------------------------------------
    % Static utility class responsible for loading oracle bone images
    % from disk into memory for use as KFS textures in the simulator.
    %
    % Responsibilities:
    %   - Scan directories for supported image formats (.png, .jpg).
    %   - Load images into MATLAB matrices using imread().
    %   - Normalize grayscale images into RGB format.
    %   - Gracefully handle corrupt or unreadable files.
    %   - Return collections of real and fake oracle bone images.
    %
    % Architectural role:
    %   - Data ingestion layer for visual assets.
    %   - Decouples file I/O from rendering logic.
    %   - Keeps simulator robust against missing assets.
    %
    % This class contains no persistent state and is fully static.
    % ===================================================================
    
    methods (Static)
        
        % ===================================================================
        % Load Oracle Bone Images
        % -------------------------------------------------------------------
        % Loads oracle bone image datasets from two directory paths:
        %
        % Inputs:
        %   realPath : Directory containing REAL oracle bone images.
        %   fakePath : Directory containing FAKE oracle bone images.
        %
        % Outputs:
        %   realImgs : Cell array of RGB image matrices.
        %   fakeImgs : Cell array of RGB image matrices.
        %
        % Supported formats:
        %   - PNG (.png)
        %   - JPEG (.jpg)
        %
        % Behavior:
        %   - Silently skips unreadable or corrupt files.
        %   - Converts grayscale images to 3-channel RGB.
        %   - Logs summary statistics after loading.
        % ===================================================================
        function [realImgs, fakeImgs] = loadOracleBones(realPath, fakePath)
            
            % Initialize output containers
            realImgs = {};
            fakeImgs = {};
            
            % ===============================================================
            % Load REAL oracle bone images
            % ===============================================================
            if ~isempty(realPath) && exist(realPath, 'dir')
                
                % Collect all PNG and JPG files in directory
                files = [ ...
                    dir(fullfile(realPath, '*.png')); ...
                    dir(fullfile(realPath, '*.jpg')) ];
                
                % Iterate over discovered files
                for f = 1:numel(files)
                    try
                        % Read image from disk
                        img = imread(fullfile(realPath, files(f).name));
                        
                        % ---------------------------------------------------
                        % Normalize grayscale → RGB
                        % Ensures renderer always receives 3-channel images.
                        % ---------------------------------------------------
                        if size(img, 3) == 1
                            img = repmat(img, [1 1 3]);
                        end
                        
                        % Append to REAL image collection
                        realImgs{end+1} = img; %#ok<AGROW>
                        
                    catch
                        % Graceful degradation for corrupt or unreadable files
                        fprintf('[WARN] Failed to load: %s\n', files(f).name);
                    end
                end
            end
            
            % ===============================================================
            % Load FAKE oracle bone images
            % ===============================================================
            if ~isempty(fakePath) && exist(fakePath, 'dir')
                
                % Collect all PNG and JPG files in directory
                files = [ ...
                    dir(fullfile(fakePath, '*.png')); ...
                    dir(fullfile(fakePath, '*.jpg')) ];
                
                % Iterate over discovered files
                for f = 1:numel(files)
                    try
                        % Read image from disk
                        img = imread(fullfile(fakePath, files(f).name));
                        
                        % Normalize grayscale → RGB
                        if size(img, 3) == 1
                            img = repmat(img, [1 1 3]);
                        end
                        
                        % Append to FAKE image collection
                        fakeImgs{end+1} = img; %#ok<AGROW>
                        
                    catch
                        % Graceful degradation for corrupt or unreadable files
                        fprintf('[WARN] Failed to load: %s\n', files(f).name);
                    end
                end
            end
            
            % ===============================================================
            % Summary Logging
            % ===============================================================
            fprintf('[OK] Loaded %d REAL, %d FAKE oracle bone images\n', ...
                numel(realImgs), numel(fakeImgs));
        end
    end
end
