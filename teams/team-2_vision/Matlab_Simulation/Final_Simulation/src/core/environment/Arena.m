classdef Arena < handle
    % ARENA Manages the forest blocks and KFS (oracle bone) placement
    
    properties
        config          % SimConfig object
        blocks          % Array of block structures
        forestMinX      % Forest boundary
        forestMaxX
        forestMinY
        forestMaxY
        
        % KFS state
        kfsIds          % Block IDs containing KFS
        kfsTypes        % "R1", "R2", or "FAKE"
        kfsImages       % Cell array of oracle bone images
        kfsColors       % Cell array of colors
        globalKfsColor  % Shared color for all KFS
        
        % Ground truth
        trueForest      % String matrix of actual KFS layout
        
        % Image libraries
        r1Images
        r2RealImages
        r2FakeImages
    end
    
    methods
        function obj = Arena(config)
            obj.config = config;
            obj.generateForest();
            obj.loadImages();
            obj.randomizeKFS();
        end
        
        function generateForest(obj)
            % Generate forest blocks with heights
            cfg = obj.config;
            
            forestWidth = cfg.FOREST_COLS * cfg.BLOCK_SIZE;
            forestHeight = cfg.FOREST_ROWS * cfg.BLOCK_SIZE;
            obj.forestMinX = (cfg.ARENA_X - forestWidth) / 2;
            obj.forestMaxX = obj.forestMinX + forestWidth;
            obj.forestMinY = (cfg.ARENA_Y - forestHeight) / 2;
            obj.forestMaxY = obj.forestMinY + forestHeight;
            
            obj.blocks = struct('id',{},'row',{},'col',{},'x',{},'y',{},'h',{});
            idCounter = 0;
            
            for r = 1:cfg.FOREST_ROWS
                for c = 1:cfg.FOREST_COLS
                    idCounter = idCounter + 1;
                    cx = obj.forestMinX + (c - 0.5) * cfg.BLOCK_SIZE;
                    cy = obj.forestMinY + (r - 0.5) * cfg.BLOCK_SIZE;
                    
                    obj.blocks(idCounter).id = idCounter;
                    obj.blocks(idCounter).row = r;
                    obj.blocks(idCounter).col = c;
                    obj.blocks(idCounter).x = cx;
                    obj.blocks(idCounter).y = cy;
                    obj.blocks(idCounter).h = cfg.heightMap(r, c);
                end
            end
        end
        
        function loadImages(obj)
            % Load oracle bone character images
            cfg = obj.config;
            fprintf('[INFO] Loading oracle bone character images...\n');
            
            [obj.r1Images, ~] = ImageLoader.loadOracleBones(cfg.getR1Folder(), []);
            [obj.r2RealImages, ~] = ImageLoader.loadOracleBones(cfg.getRealFolder(), []);
            [~, obj.r2FakeImages] = ImageLoader.loadOracleBones([], cfg.getFakeFolder());
            
            % Ensure we have at least placeholder images
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
        
        function randomizeKFS(obj)
            % Randomly place KFS on blocks
            cfg = obj.config;
            
            % Find edge and inner blocks
            edgeIds = [];
            for k = 1:numel(obj.blocks)
                b = obj.blocks(k);
                if b.row == 1 || b.row == cfg.FOREST_ROWS || ...
                   b.col == 1 || b.col == cfg.FOREST_COLS
                    edgeIds(end+1) = b.id; %#ok<AGROW>
                end
            end
            
            if numel(edgeIds) < cfg.TOTAL_R1
                error('Not enough edge blocks for R1 placement.');
            end
            
            % Place R1 on edges
            edgePerm = edgeIds(randperm(numel(edgeIds)));
            r1Ids = edgePerm(1:cfg.TOTAL_R1);
            
            % Place R2 and FAKE on remaining blocks
            remainingIds = setdiff(1:numel(obj.blocks), r1Ids);
            remainingPerm = remainingIds(randperm(numel(remainingIds)));
            
            r2RealIds = remainingPerm(1:cfg.TOTAL_R2_REAL);
            fakeIds = remainingPerm(cfg.TOTAL_R2_REAL+1 : cfg.TOTAL_R2_REAL+cfg.TOTAL_FAKE);
            
            % Store KFS info
            obj.kfsIds = [r1Ids, r2RealIds, fakeIds];
            obj.kfsTypes = [repmat("R1", 1, cfg.TOTAL_R1), ...
                           repmat("R2", 1, cfg.TOTAL_R2_REAL), ...
                           "FAKE"];
            
            % Update ground truth
            obj.trueForest = strings(cfg.FOREST_ROWS, cfg.FOREST_COLS);
            obj.trueForest(:) = "EMPTY";
            for idx = 1:numel(obj.kfsIds)
                bid = obj.kfsIds(idx);
                b = obj.blocks(bid);
                obj.trueForest(b.row, b.col) = obj.kfsTypes(idx);
            end
            
            % Assign images
            obj.kfsImages = cell(1, numel(obj.kfsIds));
            for k = 1:numel(obj.kfsIds)
                if obj.kfsTypes(k) == "R1"
                    obj.kfsImages{k} = obj.r1Images{randi(numel(obj.r1Images))};
                elseif obj.kfsTypes(k) == "R2"
                    obj.kfsImages{k} = obj.r2RealImages{randi(numel(obj.r2RealImages))};
                else
                    obj.kfsImages{k} = obj.r2FakeImages{randi(numel(obj.r2FakeImages))};
                end
            end
            
            % Assign color (shared)
            if rand < 0.5
                obj.globalKfsColor = cfg.COLOR_KFS_RED;
            else
                obj.globalKfsColor = cfg.COLOR_KFS_BLUE;
            end
            
            obj.kfsColors = cell(1, numel(obj.kfsIds));
            for k = 1:numel(obj.kfsIds)
                obj.kfsColors{k} = obj.globalKfsColor;
            end
            
            fprintf('[RANDOMIZER] New layout | R2 at: %s | R1 at: %s | Fake at: %s\n', ...
                    mat2str(r2RealIds), mat2str(r1Ids), mat2str(fakeIds));
        end
        
        function block = getBlock(obj, blockId)
            block = obj.blocks(blockId);
        end
        
        function block = getInitialBlock(obj)
            block = obj.blocks(obj.config.initialBlockId);
        end
        
        function hMap = getHeightMap(obj)
            % Return height map in grid units (20/40/60)
            cfg = obj.config;
            hMap = zeros(cfg.FOREST_ROWS, cfg.FOREST_COLS);
            for k = 1:numel(obj.blocks)
                b = obj.blocks(k);
                hMap(b.row, b.col) = b.h / 10;  % 200/400/600 -> 20/40/60
            end
        end
        
        function truth = getTrueForest(obj)
            truth = obj.trueForest;
        end
        
        function setTruthAt(obj, row, col, value)
            obj.trueForest(row, col) = value;
        end
        
        function inBounds = isInForestBounds(obj, pos2d)
            inBounds = pos2d(1) >= obj.forestMinX && pos2d(1) <= obj.forestMaxX && ...
                       pos2d(2) >= obj.forestMinY && pos2d(2) <= obj.forestMaxY;
        end
    end
end
