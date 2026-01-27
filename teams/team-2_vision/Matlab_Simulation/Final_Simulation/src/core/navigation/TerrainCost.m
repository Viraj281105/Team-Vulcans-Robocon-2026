classdef TerrainCost
    % TERRAINCOST Utility for calculating terrain-based movement costs
    
    methods (Static)
        function cost = estimate(from, to, hMap, config)
            % Estimate terrain cost using Manhattan distance + height change
            manhattan = abs(from(1) - to(1)) + abs(from(2) - to(2));
            
            h1 = hMap(from(1), from(2));
            h2 = hMap(to(1), to(2));
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
            h1 = hMap(cur(1), cur(2));
            h2 = hMap(nxt(1), nxt(2));
            dh = h2 - h1;
            
            if dh > 0
                c = (dh / 20) * config.T_UP;
            elseif dh < 0
                c = (abs(dh) / 20) * config.T_DOWN;
            else
                c = 1.0;  % Flat movement
            end
        end
    end
end
