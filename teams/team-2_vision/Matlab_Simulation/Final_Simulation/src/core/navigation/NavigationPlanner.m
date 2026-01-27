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
            % Explore UNKNOWN/UNSEEN cells with smarter prioritization
            % Prioritize: 1) Nearby cells 2) Lower terrain 3) Forward direction
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
                    
                    % Manhattan distance (primary factor)
                    dist = abs(pos(1) - r) + abs(pos(2) - c);
                    
                    % Small forward bonus (prefer exploring ahead)
                    forwardBonus = (r > pos(1)) * -0.5;
                    
                    % Small height penalty (prefer lower blocks)
                    heightPenalty = hMap(r, c) / 40.0;  % Reduced from /20
                    
                    % Confidence penalty for very low confidence
                    confPenalty = 0;
                    if conf(r, c) < 0.3
                        confPenalty = 1.0;  % Reduced from 2.0
                    end
                    
                    % Primary: minimize distance (explore nearby first!)
                    cost = dist * 2.0 + forwardBonus + heightPenalty + confPenalty;
                    
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
            
            % Check if exit blocked by R1
            for i = 1:size(exits, 1)
                e = exits(i, :);
                if belief(e(1), e(2)) == "R1" && conf(e(1), e(2)) >= config.VISION_CONFIDENCE_THRESHOLD
                    blockedR1 = e;
                    return;
                end
            end
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
