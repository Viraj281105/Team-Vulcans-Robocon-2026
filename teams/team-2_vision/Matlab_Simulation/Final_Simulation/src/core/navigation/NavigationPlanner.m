classdef NavigationPlanner
    % NAVIGATIONPLANNER Utility functions for navigation target selection
    
    methods (Static)
        function target = chooseBestR2(pos, belief, conf, carryCount, hMap, exits, timeLeft, config)
            % Hybrid cost: distance to R2 + distance R2->exit with time-aware weighting
            alpha = NavigationPlanner.interpWeight(carryCount, config.ALPHA0, config.ALPHA1, config.CAPACITY);
            beta = NavigationPlanner.interpWeight(carryCount, config.BETA0, config.BETA1, config.CAPACITY);
            
            if timeLeft < 60
                beta = beta * 2.0;  % Exit-centric late game
            end
            
            bestCost = inf;
            target = [];
            
            MIN_PLAN_CONF = 0.4;
            ROWS = size(belief, 1);
            COLS = size(belief, 2);
            
            for r = 1:ROWS
                for c = 1:COLS
                    if belief(r, c) ~= "R2" || conf(r, c) < MIN_PLAN_CONF
                        continue;
                    end
                    
                    costToR2 = TerrainCost.estimate(pos, [r c], hMap, config);
                    costToExit = min( ...
                        TerrainCost.estimate([r c], exits(1,:), hMap, config), ...
                        TerrainCost.estimate([r c], exits(2,:), hMap, config));
                    
                    totalCost = alpha * costToR2 + beta * costToExit;
                    
                    % Confidence weighting
                    totalCost = totalCost / conf(r, c);
                    
                    if totalCost < bestCost
                        bestCost = totalCost;
                        target = [r c];
                    end
                end
            end
        end
        
        function target = findBestExploreCell(pos, belief, conf, hMap, visited, config)
            % Explore UNKNOWN/UNSEEN cells ROW-BY-ROW
            bestCost = inf;
            target = [];
            
            ROWS = size(belief, 1);
            COLS = size(belief, 2);
            
            for r = 1:ROWS
                for c = 1:COLS
                    if belief(r, c) ~= "UNKNOWN" && belief(r, c) ~= "UNSEEN"
                        continue;
                    end
                    
                    if visited(r, c)
                        continue;
                    end
                    
                    % CRITICAL: Prioritize completing current row before moving forward
                    rowDist = abs(pos(1) - r);
                    
                    % MASSIVE penalty for skipping rows
                    if r < pos(1)
                        rowDist = rowDist + 100;  % Never go backward
                    elseif r > pos(1) + 1
                        rowDist = rowDist + 50;   % Can't skip rows - explore row-by-row
                    elseif r == pos(1) && belief(r, c) == "UNSEEN"
                        rowDist = rowDist - 5;    % STRONG preference for same row
                    end
                    
                    colDist = abs(pos(2) - c);
                    
                    [hRows, hCols] = size(hMap);
                    heightPenalty = 0;
                    if r >= 1 && r <= hRows && c >= 1 && c <= hCols
                        heightPenalty = hMap(r, c) / 100.0;
                    end
                    
                    % STRONG bonus for partially visible cells
                    confBonus = 0;
                    if conf(r, c) > 0.5
                        confBonus = -10.0;  % VERY strong preference
                    elseif conf(r, c) > 0.3
                        confBonus = -5.0;
                    end
                    
                    cost = (rowDist * 20.0) + colDist + heightPenalty + confBonus;
                    
                    if cost < bestCost
                        bestCost = cost;
                        target = [r c];
                    end
                end
            end
        end

        
        function [target, blockedR1] = planExitRoute(pos, belief, conf, exits, arena, config)
            % Plan route to exit, avoiding R1/FAKE obstacles
            target = [];
            blockedR1 = [];
            
            pathPlanner = PathPlanner(config, arena);
            bestPath = [];
            bestCost = inf;
            
            hMap = arena.getHeightMap();
            
            % Try pathfinding to each exit
            for i = 1:size(exits, 1)
                exitGoal = exits(i, :);
                
                % Skip if exit blocked by confirmed R1 or FAKE
                if (belief(exitGoal(1), exitGoal(2)) == "R1" || ...
                    belief(exitGoal(1), exitGoal(2)) == "FAKE") && ...
                   conf(exitGoal(1), exitGoal(2)) > 0.7
                    continue;
                end
                
                path = pathPlanner.astarWithTerrain(pos, exitGoal, belief, conf, "EXIT");
                
                if ~isempty(path)
                    pathCost = TerrainCost.computePath(path, hMap, config);
                    if pathCost < bestCost
                        bestCost = pathCost;
                        bestPath = path;
                    end
                end
            end
            
            if ~isempty(bestPath)
                target = bestPath(min(2, size(bestPath, 1)), :);
                return;
            end
            
            % FALLBACK 1: Try moving toward nearest exit (ignore obstacles temporarily)
            fprintf("[NAV DEBUG] No direct path to exit, trying movement toward exit\n");
            
            minDist = inf;
            nearestExit = [];
            for i = 1:size(exits, 1)
                dist = abs(pos(1) - exits(i,1)) + abs(pos(2) - exits(i,2));
                if dist < minDist
                    minDist = dist;
                    nearestExit = exits(i, :);
                end
            end
            
            % Move one step toward nearest exit (simple greedy)
            if ~isempty(nearestExit)
                % Try moving down (toward exit rows 4)
                if pos(1) < nearestExit(1) && pos(1) < config.FOREST_ROWS
                    target = [pos(1) + 1, pos(2)];  % Move down one row
                    fprintf("[NAV DEBUG] Moving toward exit: down to (%d,%d)\n", target);
                    return;
                end
                
                % Try moving horizontally toward exit column
                if pos(2) < nearestExit(2)
                    target = [pos(1), pos(2) + 1];  % Move right
                    fprintf("[NAV DEBUG] Moving toward exit: right to (%d,%d)\n", target);
                    return;
                elseif pos(2) > nearestExit(2)
                    target = [pos(1), pos(2) - 1];  % Move left
                    fprintf("[NAV DEBUG] Moving toward exit: left to (%d,%d)\n", target);
                    return;
                end
            end
            
            % FALLBACK 2: Check if exit blocked by R1
            for i = 1:size(exits, 1)
                e = exits(i, :);
                if belief(e(1), e(2)) == "R1" && conf(e(1), e(2)) >= config.VISION_CONFIDENCE_THRESHOLD
                    blockedR1 = e;
                    fprintf("[NAV DEBUG] Exit at (%d,%d) blocked by R1\n", e);
                    return;
                end
            end
            
            fprintf("[NAV ERROR] Could not find any exit route from (%d,%d)\n", pos);
        end

        
        function w = interpWeight(carryCount, w0, wFull, capacity)
            % Interpolate weight based on carry count
            if carryCount == 0
                w = w0;
            elseif carryCount >= capacity
                w = wFull;
            else
                w = w0 + (wFull - w0) * (carryCount / capacity);
            end
        end
    end
end
