classdef PathPlanner < handle
    % ===================================================================
    % PATHPLANNER
    % -------------------------------------------------------------------
    % Terrain-aware A* path planner operating on grid coordinates.
    %
    % Responsibilities:
    %   - Computes shortest feasible path between grid cells.
    %   - Integrates terrain elevation penalties.
    %   - Avoids obstacles using belief + confidence.
    %   - Applies mode-specific penalties (COLLECT vs EXIT).
    %
    % This class does NOT move the robot directly.
    % It purely computes optimal grid paths.
    %
    % Architectural role:
    %   - Tactical planning layer beneath Navigator.
    % ===================================================================
    
    properties
        config              % Simulation configuration parameters
        arena               % Arena reference (height map access)
    end
    
    methods
        
        % ===================================================================
        % Constructor
        % -------------------------------------------------------------------
        % Stores references required for planning.
        % ===================================================================
        function obj = PathPlanner(config, arena)
            obj.config = config;
            obj.arena  = arena;
        end
        
        
        % ===================================================================
        % A* Path Planning with Terrain Costs
        % -------------------------------------------------------------------
        % Computes a path using A* search while accounting for:
        %   - Grid boundaries
        %   - Terrain elevation costs
        %   - Dynamic obstacle belief
        %   - Mode-specific traversal penalties
        % ===================================================================
        function path = astarWithTerrain(obj, start, goal, belief, conf, currentMode)
            
            cfg  = obj.config;
            ROWS = cfg.FOREST_ROWS;
            COLS = cfg.FOREST_COLS;
            
            % Validate start and goal positions
            if ~PathPlanner.isValidCell(start, ROWS, COLS) || ...
               ~PathPlanner.isValidCell(goal,  ROWS, COLS)
                
                path = [];
                return;
            end
            
            % Retrieve terrain height map
            hMap = obj.arena.getHeightMap();
            
            % ------------------ A* State Initialization --------------------
            
            % Open set contains candidate cells to explore
            open = start;
            
            % Parent mapping for path reconstruction
            cameFrom = containers.Map;
            
            % Cost-to-come (gScore) and total estimated cost (fScore)
            gScore = inf(ROWS, COLS);
            fScore = gScore;
            
            % Initialize start node costs
            gScore(start(1), start(2)) = 0;
            fScore(start(1), start(2)) = ...
                TerrainCost.estimate(start, goal, hMap, cfg);
            
            % Safety guard to avoid infinite loops
            maxIterations = ROWS * COLS * 4;
            iterations    = 0;
            
            % ------------------ Main A* Loop -------------------------------
            while ~isempty(open) && iterations < maxIterations
                
                iterations = iterations + 1;
                
                % Select node with minimum fScore from open set
                [~, idx] = min(arrayfun( ...
                    @(k) fScore(open(k,1), open(k,2)), ...
                    1:size(open,1)));
                
                current = open(idx, :);
                open(idx, :) = [];
                
                % Goal reached → reconstruct path
                if isequal(current, goal)
                    path = PathPlanner.reconstructPath(cameFrom, current);
                    return;
                end
                
                % Generate neighboring grid cells (4-connected)
                neighbors = PathPlanner.getNeighbors(current, ROWS, COLS);
                
                % ------------------ Neighbor Expansion ---------------------
                for i = 1:size(neighbors, 1)
                    
                    neighbor = neighbors(i, :);
                    
                    % Retrieve belief and confidence
                    cellType = belief(neighbor(1), neighbor(2));
                    cellConf = conf(neighbor(1),   neighbor(2));
                    
                    % Evaluate traversability and penalty
                    [canPass, extraCost] = obj.evaluateCell( ...
                        neighbor, cellType, cellConf, ...
                        goal, currentMode);
                    
                    if ~canPass
                        continue;
                    end
                    
                    % Add pick-up cost if goal is R2
                    pickCost = 0;
                    if cellType == "R2" && isequal(neighbor, goal)
                        pickCost = cfg.T_PICK;
                    end
                    
                    % Compute tentative cost-to-come
                    tentativeG = ...
                        gScore(current(1), current(2)) + ...
                        TerrainCost.move(current, neighbor, hMap, cfg) + ...
                        extraCost + pickCost;
                    
                    % Relaxation step
                    if tentativeG < gScore(neighbor(1), neighbor(2))
                        
                        % Store parent pointer
                        cameFrom(PathPlanner.key(neighbor)) = current;
                        
                        % Update scores
                        gScore(neighbor(1), neighbor(2)) = tentativeG;
                        
                        fScore(neighbor(1), neighbor(2)) = ...
                            tentativeG + ...
                            TerrainCost.estimate(neighbor, goal, hMap, cfg);
                        
                        % Add to open set if not already present
                        if ~ismember(neighbor, open, 'rows')
                            open = [open; neighbor]; %#ok<AGROW>
                        end
                    end
                end
            end
            
            % No valid path found
            path = [];
        end
        
        
        % ===================================================================
        % Cell Evaluation Logic
        % -------------------------------------------------------------------
        % Determines whether a grid cell is traversable and computes
        % additional penalties based on belief and current mission mode.
        % ===================================================================
        function [canPass, extraCost] = evaluateCell(obj, cell, cellType, ...
                cellConf, goal, mode)
            
            cfg = obj.config;
            
            extraCost = 0;
            canPass   = true;
            
            % ------------------ Hard Constraints ---------------------------
            
            % Strongly block FAKE if confidence is high
            if cellType == "FAKE" && cellConf > 0.5
                canPass = false;
                return;
            end
            
            % Do not pass through R2 unless it is the goal cell
            if cellType == "R2" && ~isequal(cell, goal)
                canPass = false;
                return;
            end
            
            % ------------------ Soft Penalties ------------------------------
            
            % Penalize unknown low-confidence cells slightly
            if (cellType == "UNKNOWN" || cellType == "UNSEEN") && ...
                    cellConf < 0.3
                
                extraCost = extraCost + 2.0;
            end
            
            % Retrieve mode-specific penalty constants
            R1_WAIT_EXPECTED = cfg.getR1WaitExpected();
            R2_EXIT_PENALTY  = cfg.getR2ExitPenalty();
            R1_SOFT_PENALTY  = cfg.getR1SoftPenalty();
            
            % ------------------ Mode-Aware Cost Shaping ---------------------
            
            if mode == "EXIT"
                
                % Penalize R1 differently based on confidence
                if cellType == "R1"
                    
                    if cellConf > cfg.VISION_CONFIDENCE_THRESHOLD
                        extraCost = extraCost + 0.15 * R1_WAIT_EXPECTED;
                    else
                        extraCost = extraCost + 0.3 * R1_WAIT_EXPECTED;
                    end
                    
                % Discourage stepping onto R2 while exiting
                elseif cellType == "R2"
                    extraCost = extraCost + R2_EXIT_PENALTY;
                end
                
            elseif mode == "COLLECT"
                
                % Softly discourage R1 while collecting
                if cellType == "R1"
                    extraCost = extraCost + R1_SOFT_PENALTY;
                end
            end
        end
    end
    
    % ===================================================================
    % Static Utility Methods
    % ===================================================================
    methods (Static)
        
        % -------------------------------------------------------------------
        % Validates grid coordinate bounds.
        % -------------------------------------------------------------------
        function valid = isValidCell(p, rows, cols)
            valid = ...
                p(1) >= 1 && p(1) <= rows && ...
                p(2) >= 1 && p(2) <= cols;
        end
        
        
        % -------------------------------------------------------------------
        % Returns 4-connected neighbors of a grid cell.
        % -------------------------------------------------------------------
        function neighbors = getNeighbors(p, rows, cols)
            
            directions = [-1 0; 1 0; 0 -1; 0 1];
            neighbors  = [];
            
            for k = 1:4
                r = p(1) + directions(k, 1);
                c = p(2) + directions(k, 2);
                
                if r >= 1 && r <= rows && c >= 1 && c <= cols
                    neighbors = [neighbors; r c]; %#ok<AGROW>
                end
            end
        end
        
        
        % -------------------------------------------------------------------
        % Reconstructs path from parent map.
        % -------------------------------------------------------------------
        function path = reconstructPath(cameFrom, current)
            
            path = current;
            
            maxPathLength = 100;
            pathLength    = 1;
            
            % Follow parent links backwards
            while cameFrom.isKey(PathPlanner.key(current)) && ...
                  pathLength < maxPathLength
                
                current = cameFrom(PathPlanner.key(current));
                path    = [current; path]; %#ok<AGROW>
                pathLength = pathLength + 1;
            end
        end
        
        
        % -------------------------------------------------------------------
        % Generates a unique string key for map indexing.
        % -------------------------------------------------------------------
        function k = key(p)
            k = sprintf('%d,%d', p(1), p(2));
        end
    end
end
