classdef Arena < handle
    % ===================================================================
    % ARENA
    % -------------------------------------------------------------------
    % Central world model managing:
    %   - Forest block geometry and layout.
    %   - KFS (oracle bone) placement and types.
    %   - Visual textures and color assignment.
    %   - Ground-truth map used by vision and navigation logic.
    %
    % Architectural role:
    %   - Single source of truth for spatial state.
    %   - Environment authority for navigation, rendering, and vision.
    %   - Decouples geometry, textures, and semantics.
    % ===================================================================
    
    properties
        config              % SimConfig object containing constants
        
        % ------------------ Forest Geometry ------------------------------
        blocks              % Array of block structures (grid cells)
        forestMinX          % Minimum X coordinate of forest boundary
        forestMaxX          % Maximum X coordinate of forest boundary
        forestMinY          % Minimum Y coordinate of forest boundary
        forestMaxY          % Maximum Y coordinate of forest boundary
        
        % ------------------ KFS State -----------------------------------
        kfsIds              % Block IDs that contain KFS
        kfsTypes            % Corresponding KFS types ("R1", "R2", "FAKE")
        kfsImages           % Cell array of oracle bone textures
        kfsColors           % Cell array of KFS display colors
        globalKfsColor      % Shared color for all non-R1 KFS
        
        % ------------------ Ground Truth --------------------------------
        trueForest          % Matrix of actual KFS layout per grid cell
        
        % ------------------ Image Libraries -----------------------------
        r1Images            % Oracle bone images for R1 scrolls
        r2RealImages        % Oracle bone images for real R2 scrolls
        r2FakeImages        % Oracle bone images for fake scrolls
    end
    
    methods
        
        % ===================================================================
        % Constructor
        % -------------------------------------------------------------------
        % Initializes forest geometry, loads image libraries, and randomizes
        % initial KFS placement.
        % ===================================================================
        function obj = Arena(config)
            obj.config = config;
            obj.generateForest();
            obj.loadImages();
            obj.randomizeKFS();
        end
        
        
        % ===================================================================
        % Forest Generation
        % -------------------------------------------------------------------
        % Generates a rectangular grid of forest blocks centered in arena.
        % Each block receives:
        %   - Grid coordinates (row, col)
        %   - World coordinates (x, y)
        %   - Height derived from height map
        % ===================================================================
        function generateForest(obj)
            
            cfg = obj.config;
            
            % Compute forest dimensions in millimeters
            forestWidth  = cfg.FOREST_COLS * cfg.BLOCK_SIZE;
            forestHeight = cfg.FOREST_ROWS * cfg.BLOCK_SIZE;
            
            % Center forest inside arena
            obj.forestMinX = (cfg.ARENA_X - forestWidth)  / 2;
            obj.forestMaxX = obj.forestMinX + forestWidth;
            obj.forestMinY = (cfg.ARENA_Y - forestHeight) / 2;
            obj.forestMaxY = obj.forestMinY + forestHeight;
            
            % Initialize block array
            obj.blocks = struct('id',{},'row',{},'col',{},'x',{},'y',{},'h',{});
            idCounter  = 0;
            
            % Generate grid blocks
            for r = 1:cfg.FOREST_ROWS
                for c = 1:cfg.FOREST_COLS
                    
                    idCounter = idCounter + 1;
                    
                    % Compute center position of block in world coordinates
                    cx = obj.forestMinX + (c - 0.5) * cfg.BLOCK_SIZE;
                    cy = obj.forestMinY + (r - 0.5) * cfg.BLOCK_SIZE;
                    
                    % Populate block structure
                    obj.blocks(idCounter).id  = idCounter;
                    obj.blocks(idCounter).row = r;
                    obj.blocks(idCounter).col = c;
                    obj.blocks(idCounter).x   = cx;
                    obj.blocks(idCounter).y   = cy;
                    
                    % Height sourced from configuration height map
                    obj.blocks(idCounter).h   = cfg.heightMap(r, c);
                end
            end
        end
        
        
        % ===================================================================
        % Image Loading
        % -------------------------------------------------------------------
        % Loads oracle bone textures from disk using ImageLoader utility.
        % Ensures placeholder images exist if assets are missing.
        % ===================================================================
        function loadImages(obj)
            
            cfg = obj.config;
            fprintf('[INFO] Loading oracle bone character images...\n');
            
            % Load R1 textures
            [obj.r1Images, ~] = ...
                ImageLoader.loadOracleBones(cfg.getR1Folder(), []);
            
            % Load R2 REAL textures
            [obj.r2RealImages, ~] = ...
                ImageLoader.loadOracleBones(cfg.getRealFolder(), []);
            
            % Load R2 FAKE textures
            [~, obj.r2FakeImages] = ...
                ImageLoader.loadOracleBones([], cfg.getFakeFolder());
            
            % ---------------------------------------------------------------
            % Fallback placeholders ensure renderer never crashes
            % ---------------------------------------------------------------
            if isempty(obj.r1Images)
                fprintf('[WARN] No R1 images found, using placeholder\n');
                obj.r1Images = {ones(256,256,3) * 0.2};
            end
            
            if isempty(obj.r2RealImages)
                fprintf('[WARN] No R2 REAL images found, using placeholder\n');
                obj.r2RealImages = {ones(256,256,3) * 0.4};
            end
            
            if isempty(obj.r2FakeImages)
                fprintf('[WARN] No R2 FAKE images found, using placeholder\n');
                obj.r2FakeImages = {ones(256,256,3) * 0.8};
            end
        end
        
        
        % ===================================================================
        % Randomize KFS Placement
        % -------------------------------------------------------------------
        % Randomly assigns:
        %   - R1 scrolls on edge blocks only.
        %   - R2 and FAKE scrolls on remaining blocks.
        %   - Texture images and color assignments.
        %
        % Also updates the ground-truth forest map.
        % ===================================================================
        function randomizeKFS(obj)
            
            cfg = obj.config;
            
            % ---------------------------------------------------------------
            % Identify edge blocks for R1 placement
            % ---------------------------------------------------------------
            edgeIds = [];
            
            for k = 1:numel(obj.blocks)
                b = obj.blocks(k);
                
                if b.row == 1 || b.row == cfg.FOREST_ROWS || ...
                   b.col == 1 || b.col == cfg.FOREST_COLS
                   
                    edgeIds(end+1) = b.id; %#ok<AGROW>
                end
            end
            
            % Safety validation
            if numel(edgeIds) < cfg.TOTAL_R1
                error('Not enough edge blocks for R1 placement.');
            end
            
            % ---------------------------------------------------------------
            % Place R1 on randomly selected edge blocks
            % ---------------------------------------------------------------
            edgePerm = edgeIds(randperm(numel(edgeIds)));
            r1Ids    = edgePerm(1:cfg.TOTAL_R1);
            
            % ---------------------------------------------------------------
            % Place R2 and FAKE on remaining blocks
            % ---------------------------------------------------------------
            remainingIds  = setdiff(1:numel(obj.blocks), r1Ids);
            remainingPerm = remainingIds(randperm(numel(remainingIds)));
            
            r2RealIds = ...
                remainingPerm(1:cfg.TOTAL_R2_REAL);
            
            fakeIds   = ...
                remainingPerm(cfg.TOTAL_R2_REAL+1 : ...
                               cfg.TOTAL_R2_REAL+cfg.TOTAL_FAKE);
            
            % ---------------------------------------------------------------
            % Store KFS metadata
            % ---------------------------------------------------------------
            obj.kfsIds   = [r1Ids, r2RealIds, fakeIds];
            
            obj.kfsTypes = [ ...
                repmat("R1", 1, cfg.TOTAL_R1), ...
                repmat("R2", 1, cfg.TOTAL_R2_REAL), ...
                "FAKE" ];
            
            % ---------------------------------------------------------------
            % Initialize ground truth forest grid
            % ---------------------------------------------------------------
            obj.trueForest = strings(cfg.FOREST_ROWS, cfg.FOREST_COLS);
            obj.trueForest(:) = "EMPTY";
            
            for idx = 1:numel(obj.kfsIds)
                bid = obj.kfsIds(idx);
                b   = obj.blocks(bid);
                obj.trueForest(b.row, b.col) = obj.kfsTypes(idx);
            end
            
            % ---------------------------------------------------------------
            % Assign texture images to each KFS
            % ---------------------------------------------------------------
            obj.kfsImages = cell(1, numel(obj.kfsIds));
            
            for k = 1:numel(obj.kfsIds)
                
                if obj.kfsTypes(k) == "R1"
                    obj.kfsImages{k} = ...
                        obj.r1Images{randi(numel(obj.r1Images))};
                    
                elseif obj.kfsTypes(k) == "R2"
                    obj.kfsImages{k} = ...
                        obj.r2RealImages{randi(numel(obj.r2RealImages))};
                    
                else
                    obj.kfsImages{k} = ...
                        obj.r2FakeImages{randi(numel(obj.r2FakeImages))};
                end
            end
            
            % ---------------------------------------------------------------
            % Assign KFS colors
            %   - R1 scrolls are ALWAYS white.
            %   - R2/FAKE share either red or blue globally.
            % ---------------------------------------------------------------
            if rand < 0.5
                obj.globalKfsColor = cfg.COLOR_KFS_RED;
            else
                obj.globalKfsColor = cfg.COLOR_KFS_BLUE;
            end
            
            obj.kfsColors = cell(1, numel(obj.kfsIds));
            
            for k = 1:numel(obj.kfsIds)
                
                if obj.kfsTypes(k) == "R1"
                    obj.kfsColors{k} = cfg.COLOR_KFS_WHITE;
                else
                    obj.kfsColors{k} = obj.globalKfsColor;
                end
            end
            
            % ---------------------------------------------------------------
            % Logging
            % ---------------------------------------------------------------
            fprintf('[RANDOMIZER] New layout | R2 at: %s | R1 at: %s | Fake at: %s\n', ...
                mat2str(r2RealIds), mat2str(r1Ids), mat2str(fakeIds));
        end
        
        
        % ===================================================================
        % Block Accessors
        % ===================================================================
        function block = getBlock(obj, blockId)
            block = obj.blocks(blockId);
        end
        
        
        % ===================================================================
        % Initial Robot Spawn Logic
        % -------------------------------------------------------------------
        % Handles:
        %   - Normal block-based start.
        %   - Special ground start before forest entrance.
        % ===================================================================
        function block = getInitialBlock(obj)
            
            % Special case: robot starts on ground
            if obj.config.initialBlockId == -1
                
                cfg = obj.config;
                
                % First row center Y position
                firstRowY = obj.forestMinY + (0.5 * cfg.BLOCK_SIZE);
                
                % Viewing distance before forest entrance
                viewingDistance = 1500;  % mm
                
                % Populate synthetic block descriptor
                block.id  = -1;
                block.row = 0;                 % Outside grid
                block.col = 2;                 % Middle column alignment
                block.x   = obj.forestMinX + (1.5 * cfg.BLOCK_SIZE);
                block.y   = firstRowY - viewingDistance;
                block.h   = 0;                 % Ground level
                
                fprintf('[ARENA] Robot starting on GROUND at (%.0f, %.0f)\n', ...
                    block.x, block.y);
                
                fprintf('[ARENA] First row at Y=%.0fmm | Forest starts at Y=%.0fmm\n', ...
                    firstRowY, obj.forestMinY);
                
                fprintf('[ARENA] Viewing distance: %.0fmm | Can see all 3 blocks in row 1\n', ...
                    viewingDistance);
                
            else
                % Normal case: start on a specific block
                block = obj.blocks(obj.config.initialBlockId);
                
                fprintf('[ARENA] Robot starting on BLOCK %d at (%.0f, %.0f)\n', ...
                    block.id, block.x, block.y);
            end
        end

        
        % ===================================================================
        % Height Map Access
        % -------------------------------------------------------------------
        % Converts block heights into grid height map scaled for planners.
        % ===================================================================
        function hMap = getHeightMap(obj)
            
            cfg  = obj.config;
            hMap = zeros(cfg.FOREST_ROWS, cfg.FOREST_COLS);
            
            for k = 1:numel(obj.blocks)
                b = obj.blocks(k);
                
                % Convert mm heights to scaled grid units
                hMap(b.row, b.col) = b.h / 10;   % 200/400/600 → 20/40/60
            end
        end
        
        
        % ===================================================================
        % Ground Truth Accessors
        % ===================================================================
        function truth = getTrueForest(obj)
            truth = obj.trueForest;
        end
        
        function setTruthAt(obj, row, col, value)
            obj.trueForest(row, col) = value;
        end
        
        
        % ===================================================================
        % Forest Boundary Check
        % -------------------------------------------------------------------
        % Validates whether a 2D position lies inside forest boundaries.
        % ===================================================================
        function inBounds = isInForestBounds(obj, pos2d)
            
            inBounds = ...
                pos2d(1) >= obj.forestMinX && ...
                pos2d(1) <= obj.forestMaxX && ...
                pos2d(2) >= obj.forestMinY && ...
                pos2d(2) <= obj.forestMaxY;
        end
    end
end
