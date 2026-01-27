classdef TerrainCost
    % TERRAINCOST Utility for calculating terrain-based movement costs
    % FIXED: Handles positions outside grid bounds (ground start)
    
    methods (Static)
        function cost = estimate(from, to, hMap, config)
            % Estimate terrain cost using Manhattan distance + height change
            % Handles ground start positions outside grid
            
            manhattan = abs(from(1) - to(1)) + abs(from(2) - to(2));
            
            % Check if positions are within hMap bounds
            [rows, cols] = size(hMap);
            
            % Get heights (default to 0 if outside bounds)
            h1 = TerrainCost.getHeight(from, hMap, rows, cols);
            h2 = TerrainCost.getHeight(to, hMap, rows, cols);
            
            dh = h2 - h1;
            
            if dh > 0
                heightCost = (dh / 20) * config.T_UP;
            elseif dh < 0
                heightCost = (abs(dh) / 20) * config.T_DOWN;
            else
                heightCost = 0;
            end
            
            cost = manhattan + heightCost;
        end
        
        function cost = computePath(path, hMap, config)
            % Compute total cost of a path
            cost = 0;
            for i = 1:size(path, 1) - 1
                cur = path(i, :);
                nxt = path(i+1, :);
                cost = cost + TerrainCost.move(cur, nxt, hMap, config);
            end
        end
        
        function c = move(cur, nxt, hMap, config)
            % Calculate cost of moving from cur to nxt cell
            % Handles ground start positions outside grid
            
            [rows, cols] = size(hMap);
            
            % Get heights (default to 0 if outside bounds)
            h1 = TerrainCost.getHeight(cur, hMap, rows, cols);
            h2 = TerrainCost.getHeight(nxt, hMap, rows, cols);
            
            dh = h2 - h1;
            
            if dh > 0
                c = (dh / 20) * config.T_UP;
            elseif dh < 0
                c = (abs(dh) / 20) * config.T_DOWN;
            else
                c = 1.0;  % Flat movement
            end
        end
        
        function h = getHeight(pos, hMap, rows, cols)
            % NEW: Helper to safely get height with bounds checking
            if pos(1) >= 1 && pos(1) <= rows && pos(2) >= 1 && pos(2) <= cols
                h = hMap(pos(1), pos(2));
            else
                h = 0;  % Ground level (outside forest)
            end
        end
    end
end
