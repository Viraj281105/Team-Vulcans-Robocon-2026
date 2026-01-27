classdef PathPlanner < handle
    % PATHPLANNER A* pathfinding with terrain-aware costs
    
    properties
        config
        arena
    end
    
    methods
        function obj = PathPlanner(config, arena)
            obj.config = config;
            obj.arena = arena;
        end
        
        function path = astarWithTerrain(obj, start, goal, belief, conf, currentMode)
            % A* pathfinding with terrain costs and obstacle avoidance
            cfg = obj.config;
            ROWS = cfg.FOREST_ROWS;
            COLS = cfg.FOREST_COLS;
            
            if ~PathPlanner.isValidCell(start, ROWS, COLS) || ...
               ~PathPlanner.isValidCell(goal, ROWS, COLS)
                path = [];
                return;
            end
            
            hMap = obj.arena.getHeightMap();
            
            open = start;
            cameFrom = containers.Map;
            gScore = inf(ROWS, COLS);
            fScore = gScore;
            
            gScore(start(1), start(2)) = 0;
            fScore(start(1), start(2)) = TerrainCost.estimate(start, goal, hMap, cfg);
            
            maxIterations = ROWS * COLS * 4;
            iterations = 0;
            
            while ~isempty(open) && iterations < maxIterations
                iterations = iterations + 1;
                
                % Find cell with lowest fScore
                [~, idx] = min(arrayfun(@(k) fScore(open(k,1), open(k,2)), 1:size(open,1)));
                current = open(idx, :);
                open(idx, :) = [];
                
                if isequal(current, goal)
                    path = PathPlanner.reconstructPath(cameFrom, current);
                    return;
                end
                
                neighbors = PathPlanner.getNeighbors(current, ROWS, COLS);
                
                for i = 1:size(neighbors, 1)
                    neighbor = neighbors(i, :);
                    cellType = belief(neighbor(1), neighbor(2));
                    cellConf = conf(neighbor(1), neighbor(2));
                    
                    [canPass, extraCost] = obj.evaluateCell(neighbor, cellType, ...
                        cellConf, goal, currentMode);
                    
                    if ~canPass
                        continue;
                    end
                    
                    % Add pick cost if ending on R2
                    pickCost = 0;
                    if cellType == "R2" && isequal(neighbor, goal)
                        pickCost = cfg.T_PICK;
                    end
                    
                    tentativeG = gScore(current(1), current(2)) + ...
                                TerrainCost.move(current, neighbor, hMap, cfg) + ...
                                extraCost + pickCost;
                    
                    if tentativeG < gScore(neighbor(1), neighbor(2))
                        cameFrom(PathPlanner.key(neighbor)) = current;
                        gScore(neighbor(1), neighbor(2)) = tentativeG;
                        fScore(neighbor(1), neighbor(2)) = tentativeG + ...
                            TerrainCost.estimate(neighbor, goal, hMap, cfg);
                        
                        if ~ismember(neighbor, open, 'rows')
                            open = [open; neighbor]; %#ok<AGROW>
                        end
                    end
                end
            end
            
            path = [];
        end
        
        function [canPass, extraCost] = evaluateCell(obj, cell, cellType, cellConf, goal, mode)
            % Evaluate if a cell can be passed and compute extra cost
            cfg = obj.config;
            extraCost = 0;
            canPass = true;
            
            % Strongly avoid FAKE if confident
            if cellType == "FAKE" && cellConf > 0.5
                canPass = false;
                return;
            end
            
            % Can't pass through R2 unless it's the goal
            if cellType == "R2" && ~isequal(cell, goal)
                canPass = false;
                return;
            end
            
            % Unknown cells get slight penalty if very low confidence
            if (cellType == "UNKNOWN" || cellType == "UNSEEN") && cellConf < 0.3
                extraCost = extraCost + 2.0;
            end
            
            % Mode-specific penalties
            R1_WAIT_EXPECTED = cfg.getR1WaitExpected();
            R2_EXIT_PENALTY = cfg.getR2ExitPenalty();
            R1_SOFT_PENALTY = cfg.getR1SoftPenalty();
            
            if mode == "EXIT"
                if cellType == "R1"
                    if cellConf > cfg.VISION_CONFIDENCE_THRESHOLD
                        extraCost = extraCost + 0.15 * R1_WAIT_EXPECTED;
                    else
                        extraCost = extraCost + 0.3 * R1_WAIT_EXPECTED;
                    end
                elseif cellType == "R2"
                    extraCost = extraCost + R2_EXIT_PENALTY;
                end
            elseif mode == "COLLECT"
                if cellType == "R1"
                    extraCost = extraCost + R1_SOFT_PENALTY;
                end
            end
        end
    end
    
    methods (Static)
        function valid = isValidCell(p, rows, cols)
            valid = p(1) >= 1 && p(1) <= rows && p(2) >= 1 && p(2) <= cols;
        end
        
        function neighbors = getNeighbors(p, rows, cols)
            directions = [-1 0; 1 0; 0 -1; 0 1];
            neighbors = [];
            
            for k = 1:4
                r = p(1) + directions(k, 1);
                c = p(2) + directions(k, 2);
                if r >= 1 && r <= rows && c >= 1 && c <= cols
                    neighbors = [neighbors; r c]; %#ok<AGROW>
                end
            end
        end
        
        function path = reconstructPath(cameFrom, current)
            path = current;
            maxPathLength = 100;
            pathLength = 1;
            
            while cameFrom.isKey(PathPlanner.key(current)) && pathLength < maxPathLength
                current = cameFrom(PathPlanner.key(current));
                path = [current; path]; %#ok<AGROW>
                pathLength = pathLength + 1;
            end
        end
        
        function k = key(p)
            k = sprintf('%d,%d', p(1), p(2));
        end
    end
end
