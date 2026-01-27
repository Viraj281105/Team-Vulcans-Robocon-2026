classdef NavigationPlanner
    % ===================================================================
    % NAVIGATIONPLANNER
    % -------------------------------------------------------------------
    % Static utility class for high-level navigation target selection.
    %
    % Responsibilities:
    %   - Decide which R2 target should be collected next.
    %   - Select exploration targets for unknown areas.
    %   - Plan safe exit routing under uncertainty and obstacles.
    %   - Adapt decision weights based on carry load and time pressure.
    %
    % Architectural role:
    %   - Strategic planner layer (goal selection).
    %   - Operates ABOVE low-level path planners.
    %   - Consumes belief grids, confidence maps, terrain costs.
    %
    % This class is intentionally stateless and deterministic.
    % ===================================================================
    
    methods (Static)
        
        % ===================================================================
        % Choose Best R2 Target
        % -------------------------------------------------------------------
        % Selects the optimal R2 cell using a hybrid weighted cost model:
        %
        %   TotalCost =
        %       alpha * (cost from robot → R2) +
        %       beta  * (cost from R2 → nearest exit)
        %
        % The weights alpha and beta are interpolated based on:
        %   - How many items are currently being carried.
        %   - Remaining capacity.
        %   - Remaining mission time (late-game exit bias).
        %
        % Additional scaling:
        %   - Divides by confidence to favor reliable detections.
        % ===================================================================
        function target = chooseBestR2( ...
                pos, belief, conf, carryCount, ...
                hMap, exits, timeLeft, config)
            
            % Interpolate cost weights based on carry load
            alpha = NavigationPlanner.interpWeight( ...
                carryCount, config.ALPHA0, ...
                config.ALPHA1, config.CAPACITY);
            
            beta  = NavigationPlanner.interpWeight( ...
                carryCount, config.BETA0, ...
                config.BETA1, config.CAPACITY);
            
            % Late-game bias toward exits when time is low
            if timeLeft < 60
                beta = beta * 2.0;
            end
            
            % Initialize search state
            bestCost = inf;
            target   = [];
            
            % Minimum confidence threshold for planning
            MIN_PLAN_CONF = 0.4;
            
            ROWS = size(belief, 1);
            COLS = size(belief, 2);
            
            % ---------------------------------------------------------------
            % Exhaustive scan of belief grid
            % ---------------------------------------------------------------
            for r = 1:ROWS
                for c = 1:COLS
                    
                    % Only consider high-confidence R2 candidates
                    if belief(r, c) ~= "R2" || ...
                       conf(r, c) < MIN_PLAN_CONF
                        continue;
                    end
                    
                    % Cost from current robot position to R2 cell
                    costToR2 = TerrainCost.estimate( ...
                        pos, [r c], hMap, config);
                    
                    % Cost from R2 cell to nearest exit
                    costToExit = min( ...
                        TerrainCost.estimate( ...
                            [r c], exits(1,:), hMap, config), ...
                        TerrainCost.estimate( ...
                            [r c], exits(2,:), hMap, config));
                    
                    % Hybrid weighted cost
                    totalCost = alpha * costToR2 + ...
                                beta  * costToExit;
                    
                    % Confidence scaling (lower confidence → higher cost)
                    totalCost = totalCost / conf(r, c);
                    
                    % Track best candidate
                    if totalCost < bestCost
                        bestCost = totalCost;
                        target   = [r c];
                    end
                end
            end
        end
        
        
        % ===================================================================
        % Exploration Target Selection
        % -------------------------------------------------------------------
        % Chooses the next grid cell to explore when no confident R2 exists.
        %
        % Strategy:
        %   - Explore row-by-row systematically.
        %   - Penalize backward movement heavily.
        %   - Avoid skipping rows prematurely.
        %   - Favor partially observed cells.
        %   - Add height penalty to avoid excessive climbing.
        % ===================================================================
        function target = findBestExploreCell( ...
                pos, belief, conf, hMap, visited, config)
            
            bestCost = inf;
            target   = [];
            
            ROWS = size(belief, 1);
            COLS = size(belief, 2);
            
            % ---------------------------------------------------------------
            % Scan entire grid for candidate exploration cells
            % ---------------------------------------------------------------
            for r = 1:ROWS
                for c = 1:COLS
                    
                    % Only explore UNKNOWN or UNSEEN cells
                    if belief(r, c) ~= "UNKNOWN" && ...
                       belief(r, c) ~= "UNSEEN"
                        continue;
                    end
                    
                    % Skip already visited cells
                    if visited(r, c)
                        continue;
                    end
                    
                    % -------------------------------------------------------
                    % Row progression logic
                    % -------------------------------------------------------
                    rowDist = abs(pos(1) - r);
                    
                    % Massive penalties for undesirable row transitions
                    if r < pos(1)
                        rowDist = rowDist + 100;     % Never go backward
                    elseif r > pos(1) + 1
                        rowDist = rowDist + 50;      % Avoid skipping rows
                    elseif r == pos(1) && ...
                           belief(r, c) == "UNSEEN"
                        rowDist = rowDist - 5;       % Favor same row
                    end
                    
                    % Column distance penalty
                    colDist = abs(pos(2) - c);
                    
                    % -------------------------------------------------------
                    % Height penalty from height map
                    % -------------------------------------------------------
                    [hRows, hCols] = size(hMap);
                    heightPenalty = 0;
                    
                    if r >= 1 && r <= hRows && ...
                       c >= 1 && c <= hCols
                        heightPenalty = hMap(r, c) / 100.0;
                    end
                    
                    % -------------------------------------------------------
                    % Confidence bonus encourages partial observations
                    % -------------------------------------------------------
                    confBonus = 0;
                    
                    if conf(r, c) > 0.5
                        confBonus = -10.0;
                    elseif conf(r, c) > 0.3
                        confBonus = -5.0;
                    end
                    
                    % Composite heuristic cost
                    cost = (rowDist * 20.0) + ...
                           colDist + ...
                           heightPenalty + ...
                           confBonus;
                    
                    % Track best candidate
                    if cost < bestCost
                        bestCost = cost;
                        target   = [r c];
                    end
                end
            end
        end

        
        % ===================================================================
        % Exit Route Planning
        % -------------------------------------------------------------------
        % Attempts to compute a safe path toward an exit while:
        %   - Avoiding confirmed R1 and FAKE obstacles.
        %   - Falling back to greedy motion if planner fails.
        %   - Detecting blocked exits.
        % ===================================================================
        function [target, blockedR1] = planExitRoute( ...
                pos, belief, conf, exits, arena, config)
            
            target     = [];
            blockedR1  = [];
            
            % Instantiate path planner instance
            pathPlanner = PathPlanner(config, arena);
            
            bestPath = [];
            bestCost = inf;
            
            % Height map for terrain cost evaluation
            hMap = arena.getHeightMap();
            
            % ---------------------------------------------------------------
            % Attempt pathfinding toward each exit
            % ---------------------------------------------------------------
            for i = 1:size(exits, 1)
                
                exitGoal = exits(i, :);
                
                % Skip exits blocked by high-confidence R1 or FAKE
                if (belief(exitGoal(1), exitGoal(2)) == "R1" || ...
                    belief(exitGoal(1), exitGoal(2)) == "FAKE") && ...
                   conf(exitGoal(1), exitGoal(2)) > 0.7
                    continue;
                end
                
                % Compute path using terrain-aware A*
                path = pathPlanner.astarWithTerrain( ...
                    pos, exitGoal, belief, conf, "EXIT");
                
                % Evaluate path cost if valid
                if ~isempty(path)
                    pathCost = ...
                        TerrainCost.computePath(path, hMap, config);
                    
                    if pathCost < bestCost
                        bestCost = pathCost;
                        bestPath = path;
                    end
                end
            end
            
            % ---------------------------------------------------------------
            % If a valid path exists, return next step
            % ---------------------------------------------------------------
            if ~isempty(bestPath)
                target = bestPath(min(2, size(bestPath, 1)), :);
                return;
            end
            
            % ---------------------------------------------------------------
            % FALLBACK 1:
            % Greedy movement toward nearest exit
            % ---------------------------------------------------------------
            fprintf("[NAV DEBUG] No direct path to exit, trying movement toward exit\n");
            
            minDist      = inf;
            nearestExit  = [];
            
            for i = 1:size(exits, 1)
                dist = ...
                    abs(pos(1) - exits(i,1)) + ...
                    abs(pos(2) - exits(i,2));
                
                if dist < minDist
                    minDist     = dist;
                    nearestExit = exits(i, :);
                end
            end
            
            % Move one step greedily toward nearest exit
            if ~isempty(nearestExit)
                
                % Try moving downward toward exit rows
                if pos(1) < nearestExit(1) && ...
                   pos(1) < config.FOREST_ROWS
                    
                    target = [pos(1) + 1, pos(2)];
                    fprintf("[NAV DEBUG] Moving toward exit: down to (%d,%d)\n", target);
                    return;
                end
                
                % Try horizontal adjustment
                if pos(2) < nearestExit(2)
                    target = [pos(1), pos(2) + 1];
                    fprintf("[NAV DEBUG] Moving toward exit: right to (%d,%d)\n", target);
                    return;
                    
                elseif pos(2) > nearestExit(2)
                    target = [pos(1), pos(2) - 1];
                    fprintf("[NAV DEBUG] Moving toward exit: left to (%d,%d)\n", target);
                    return;
                end
            end
            
            % ---------------------------------------------------------------
            % FALLBACK 2:
            % Detect if exit is blocked by confirmed R1
            % ---------------------------------------------------------------
            for i = 1:size(exits, 1)
                
                e = exits(i, :);
                
                if belief(e(1), e(2)) == "R1" && ...
                   conf(e(1), e(2)) >= ...
                        config.VISION_CONFIDENCE_THRESHOLD
                    
                    blockedR1 = e;
                    fprintf("[NAV DEBUG] Exit at (%d,%d) blocked by R1\n", e);
                    return;
                end
            end
            
            % ---------------------------------------------------------------
            % Failure condition
            % ---------------------------------------------------------------
            fprintf("[NAV ERROR] Could not find any exit route from (%d,%d)\n", pos);
        end

        
        % ===================================================================
        % Weight Interpolation Utility
        % -------------------------------------------------------------------
        % Linearly interpolates weights based on carry load.
        % ===================================================================
        function w = interpWeight(carryCount, w0, wFull, capacity)
            
            if carryCount == 0
                w = w0;
                
            elseif carryCount >= capacity
                w = wFull;
                
            else
                w = w0 + ...
                    (wFull - w0) * (carryCount / capacity);
            end
        end
    end
end
