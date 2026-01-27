classdef TerrainCost
    % ===================================================================
    % TERRAINCOST
    % -------------------------------------------------------------------
    % Static utility class for estimating movement costs across terrain.
    %
    % Purpose:
    %   - Converts grid transitions and elevation changes into scalar costs.
    %   - Used by path planners (Hybrid A*, grid planners, heuristics).
    %   - Penalizes uphill motion more than downhill motion.
    %   - Safely handles positions outside the height grid (ground start).
    %
    % Design intent:
    %   - Stateless and deterministic.
    %   - Fully reusable across planners and simulations.
    %   - Explicit bounds checking to prevent indexing errors.
    %
    % This class intentionally contains no simulation state.
    % ===================================================================
    
    methods (Static)
        
        % ===================================================================
        % Heuristic Cost Estimation
        % -------------------------------------------------------------------
        % Estimates movement cost from one grid cell to another using:
        %   - Manhattan distance in grid space.
        %   - Height change penalty derived from height map.
        %
        % Handles cases where start or goal lies outside grid bounds
        % (e.g., ground start positions).
        % ===================================================================
        function cost = estimate(from, to, hMap, config)
            
            % Manhattan distance heuristic in grid coordinates
            manhattan = ...
                abs(from(1) - to(1)) + abs(from(2) - to(2));
            
            % Dimensions of height map grid
            [rows, cols] = size(hMap);
            
            % ---------------------------------------------------------------
            % Retrieve terrain heights safely
            % Defaults to 0 when outside grid bounds.
            % ---------------------------------------------------------------
            h1 = TerrainCost.getHeight(from, hMap, rows, cols);
            h2 = TerrainCost.getHeight(to,   hMap, rows, cols);
            
            % Height difference between cells
            dh = h2 - h1;
            
            % ---------------------------------------------------------------
            % Height penalty model
            %   - Uphill movement is more expensive.
            %   - Downhill movement has smaller penalty.
            %   - Flat movement has zero height cost.
            % ---------------------------------------------------------------
            if dh > 0
                heightCost = (dh / 20) * config.T_UP;
            elseif dh < 0
                heightCost = (abs(dh) / 20) * config.T_DOWN;
            else
                heightCost = 0;
            end
            
            % Final heuristic cost
            cost = manhattan + heightCost;
        end
        
        
        % ===================================================================
        % Path Cost Aggregation
        % -------------------------------------------------------------------
        % Computes cumulative cost of an entire path sequence.
        % ===================================================================
        function cost = computePath(path, hMap, config)
            
            cost = 0;
            
            % Accumulate cost between consecutive path nodes
            for i = 1:size(path, 1) - 1
                cur = path(i,   :);
                nxt = path(i+1, :);
                cost = cost + ...
                    TerrainCost.move(cur, nxt, hMap, config);
            end
        end
        
        
        % ===================================================================
        % Single-Step Movement Cost
        % -------------------------------------------------------------------
        % Calculates cost of moving from one grid cell to the next.
        % ===================================================================
        function c = move(cur, nxt, hMap, config)
            
            % Dimensions of height grid
            [rows, cols] = size(hMap);
            
            % Retrieve heights safely with bounds checking
            h1 = TerrainCost.getHeight(cur, hMap, rows, cols);
            h2 = TerrainCost.getHeight(nxt, hMap, rows, cols);
            
            % Height difference
            dh = h2 - h1;
            
            % Apply asymmetric height penalty
            if dh > 0
                c = (dh / 20) * config.T_UP;
            elseif dh < 0
                c = (abs(dh) / 20) * config.T_DOWN;
            else
                c = 1.0;     % Flat terrain movement baseline cost
            end
        end
        
        
        % ===================================================================
        % Height Lookup Helper
        % -------------------------------------------------------------------
        % Safely retrieves height from grid with bounds checking.
        % ===================================================================
        function h = getHeight(pos, hMap, rows, cols)
            
            % Check whether indices are inside grid boundaries
            if pos(1) >= 1 && pos(1) <= rows && ...
               pos(2) >= 1 && pos(2) <= cols
                
                h = hMap(pos(1), pos(2));
            else
                % Outside forest bounds → treated as ground level
                h = 0;
            end
        end
    end
end
