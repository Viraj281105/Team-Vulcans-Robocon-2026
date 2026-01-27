classdef Navigator < handle
    % NAVIGATOR Time-aware hybrid A* path planner for autonomous navigation
    
    properties
        config
        arena
        
        % Navigation state
        mode            % "COLLECT", "EXIT", or "WAIT_FOR_EXIT"
        carry           % Number of items carried
        position        % Current grid position [row, col]
        
        % Belief state
        belief          % Grid belief map
        beliefConfidence
        visitedCells
        
        % Path planning
        pathHistory
        
        % Timing
        gameStartTime
        emergencyExitTriggered
        
        % R1 waiting
        r1WaitStart
        r1WaitAttempts
        
        % Stuck detection
        lastGridPos
        stuckCounter
        replanCounter
        
        % Collection tracking
        collectedR2Positions
        pathFailureCount
        
        % Mission completion
        missionComplete
    end
    
    properties (Constant)
        POSITION_TOLERANCE = 0.1
        MAX_STUCK_COUNT = 3
        MAX_REPLAN_ATTEMPTS = 5
        MIN_PLAN_CONF = 0.4
    end
    
    methods
        function obj = Navigator(config, arena)
            obj.config = config;
            obj.arena = arena;
            obj.mode = "COLLECT";
            obj.carry = 0;
            
            initialBlock = arena.getInitialBlock();
            obj.position = [initialBlock.row, initialBlock.col];
            
            obj.belief = strings(config.FOREST_ROWS, config.FOREST_COLS);
            obj.belief(:) = "UNSEEN";
            obj.beliefConfidence = zeros(config.FOREST_ROWS, config.FOREST_COLS);
            
            obj.visitedCells = false(config.FOREST_ROWS, config.FOREST_COLS);
            obj.visitedCells(obj.position(1), obj.position(2)) = true;
            
            obj.pathHistory = obj.position;
            obj.gameStartTime = tic;
            obj.emergencyExitTriggered = false;
            
            obj.r1WaitStart = containers.Map;
            obj.r1WaitAttempts = containers.Map;
            
            obj.lastGridPos = obj.position;
            obj.stuckCounter = 0;
            obj.replanCounter = 0;
            
            obj.collectedR2Positions = [];
            obj.pathFailureCount = 0;
            obj.missionComplete = false;
        end
        
        function step(obj, robot, arena, vision, detections)
            % Execute one autonomous navigation step
            cfg = obj.config;
            
            % Update grid position from robot
            obj.position = robot.gridPos;
            
            % Check time limits
            elapsedTime = toc(obj.gameStartTime);
            remainingTime = cfg.TIME_LIMIT - elapsedTime;
            
            if remainingTime < cfg.TIME_BUFFER && ~obj.emergencyExitTriggered
                fprintf("[NAV] EMERGENCY: Only %.1fs left → EXIT mode\n", remainingTime);
                obj.mode = "EXIT";
                obj.emergencyExitTriggered = true;
                obj.carry = max(obj.carry, 1);
            end
            
            if remainingTime <= 0
                fprintf("[NAV] TIMEOUT, stopping autonomous\n");
                return;
            end
            
            % Update belief from ALL detected KFS (optimization)
            obj.updateBeliefFromDetections(vision, arena);
            
            % Update belief from current vision
            robotState = robot.getState();
            cameraPos = robotState.position + [0 0 cfg.CAMERA_HEIGHT_OFFSET];
            cameraPitch = 15;  % Would come from camera controller
            
            [obj.belief, obj.beliefConfidence] = vision.updateGridBelief(...
                cameraPos, robotState.yaw, cameraPitch, obj.belief, obj.beliefConfidence);
            
            obj.visitedCells(obj.position(1), obj.position(2)) = true;
            
            % Stuck detection
            obj.detectStuck();
            
            % Immediate pick if on R2
            if obj.tryPickKFS(arena)
                return;
            end
            
            % Check if at exit
            if obj.checkExitReached()
                return;
            end
            
            % R1 waiting logic
            if obj.mode == "WAIT_FOR_EXIT"
                obj.handleR1Waiting(arena);
                return;
            end
            
            % Plan and execute move
            obj.planAndMove(robot, arena, remainingTime);
        end
        
        function detectStuck(obj)
            if norm(obj.position - obj.lastGridPos) < obj.POSITION_TOLERANCE
                obj.stuckCounter = obj.stuckCounter + 1;
                if obj.stuckCounter >= obj.MAX_STUCK_COUNT
                    fprintf("[NAV] Stuck, re-evaluating...\n");
                    if obj.mode == "COLLECT" && obj.carry > 0
                        obj.mode = "EXIT";
                    end
                    obj.stuckCounter = 0;
                    obj.replanCounter = obj.replanCounter + 1;
                    
                    if obj.replanCounter >= obj.MAX_REPLAN_ATTEMPTS
                        fprintf("[NAV] Max replans reached, forcing EXIT\n");
                        obj.mode = "EXIT";
                        obj.emergencyExitTriggered = true;
                    end
                end
            else
                obj.stuckCounter = 0;
                obj.lastGridPos = obj.position;
            end
        end
        
        function picked = tryPickKFS(obj, arena)
            picked = false;
            cfg = obj.config;
            
            if obj.belief(obj.position(1), obj.position(2)) == "R2" && ...
               obj.beliefConfidence(obj.position(1), obj.position(2)) >= cfg.VISION_CONFIDENCE_THRESHOLD
                
                obj.carry = obj.carry + 1;
                arena.setTruthAt(obj.position(1), obj.position(2), "EMPTY");
                obj.belief(obj.position(1), obj.position(2)) = "EMPTY";
                obj.collectedR2Positions = [obj.collectedR2Positions; obj.position];
                
                fprintf("[NAV] PICK at (%d,%d), carry=%d/%d\n", ...
                    obj.position, obj.carry, cfg.CAPACITY);
                
                if obj.carry >= cfg.CAPACITY
                    obj.mode = "EXIT";
                    fprintf("[NAV] Capacity full → EXIT\n");
                end
                
                picked = true;
            end
        end
        
        function reached = checkExitReached(obj)
            reached = false;
            if ismember(obj.position, obj.config.exitCells, 'rows')
                if obj.mode == "EXIT" || obj.carry >= obj.config.CAPACITY
                    fprintf("[NAV] ✓ MISSION COMPLETE! Reached exit at (%d,%d) with %d items\n", ...
                        obj.position, obj.carry);
                    obj.missionComplete = true;
                    reached = true;
                end
            end
        end
        
        function handleR1Waiting(obj, arena)
            cfg = obj.config;
            keysR1 = obj.r1WaitStart.keys;
            
            if isempty(keysR1)
                obj.mode = "EXIT";
                return;
            end
            
            kR1 = keysR1{1};
            elapsedR1 = toc(obj.r1WaitStart(kR1));
            
            if elapsedR1 >= cfg.WAIT_TIME
                if rand <= cfg.R1_CLEAR_PROB
                    rc = sscanf(kR1, '%d,%d');
                    arena.setTruthAt(rc(1), rc(2), "EMPTY");
                    obj.belief(rc(1), rc(2)) = "EMPTY";
                    obj.r1WaitStart.remove(kR1);
                    fprintf("[NAV] R1 CLEARED at (%d,%d)\n", rc);
                    obj.mode = "EXIT";
                else
                    obj.r1WaitStart(kR1) = tic;
                    fprintf("[NAV] R1 NOT CLEARED → RETRY WAIT\n");
                end
            else
                fprintf("[NAV] WAITING FOR R1 (%.1fs / %ds)\n", elapsedR1, cfg.WAIT_TIME);
            end
        end
        
        function planAndMove(obj, robot, arena, remainingTime)
            target = [];
            
            if obj.mode == "COLLECT"
                target = obj.chooseBestR2(remainingTime);
                
                if isempty(target)
                    % Count how many R2s we know about
                    r2Count = sum(obj.belief == "R2", 'all');
                    if r2Count > 0
                        fprintf("[NAV DEBUG] %d R2(s) in belief but none selected (confidence too low?)\n", r2Count);
                    end
                    
                    target = obj.findBestExploreCell();
                    if isempty(target)
                        if obj.carry > 0
                            fprintf("[NAV] NO R2 & NO UNKNOWN → EXIT MODE\n");
                            obj.mode = "EXIT";
                        end
                        return;
                    else
                        fprintf("[NAV] Exploring cell (%d,%d)\n", target);
                    end
                else
                    fprintf("[NAV] Target R2 at (%d,%d)\n", target);
                end
                
            elseif obj.mode == "EXIT"
                [target, blockedR1] = obj.planExitRoute();
                
                if isempty(target) && ~isempty(blockedR1)
                    r1key = sprintf('%d,%d', blockedR1(1), blockedR1(2));
                    if ~obj.r1WaitStart.isKey(r1key)
                        obj.r1WaitStart(r1key) = tic;
                        fprintf("[NAV] EXIT BLOCKED by R1 → WAITING\n");
                    end
                    obj.mode = "WAIT_FOR_EXIT";
                    return;
                elseif isempty(target)
                    fprintf("[NAV] NO VALID EXIT PATH\n");
                    return;
                else
                    fprintf("[NAV] Exit route via (%d,%d)\n", target);
                end
            end
            
            % Execute move
            if ~isempty(target)
                obj.executeMove(target, robot, arena);
            end
        end
        
        function executeMove(obj, target, robot, arena)
            pathPlanner = PathPlanner(obj.config, obj.arena);
            path = pathPlanner.astarWithTerrain(obj.position, target, ...
                obj.belief, obj.beliefConfidence, obj.mode);
            
            if isempty(path) || size(path, 1) < 2
                fprintf("[NAV] No path to (%d,%d)\n", target);
                return;
            end
            
            nextCell = path(2, :);
            
            % Debug output
            fprintf("[NAV DEBUG] Current: (%d,%d), Target: (%d,%d), Next: (%d,%d), PathLen: %d\n", ...
                obj.position(1), obj.position(2), target(1), target(2), ...
                nextCell(1), nextCell(2), size(path,1));
            
            if abs(nextCell(1) - obj.position(1)) + abs(nextCell(2) - obj.position(2)) ~= 1
                fprintf("[NAV] Non-adjacent step, skipping\n");
                return;
            end
            
            % Find block by row/col (not using sub2ind which assumes wrong ordering)
            bNext = [];
            for k = 1:numel(obj.arena.blocks)
                b = obj.arena.blocks(k);
                if b.row == nextCell(1) && b.col == nextCell(2)
                    bNext = b;
                    break;
                end
            end
            
            if isempty(bNext)
                fprintf("[NAV ERROR] Could not find block at grid (%d,%d)\n", nextCell);
                return;
            end
            
            % Update robot 3D position
            robot.moveTo([bNext.x, bNext.y], obj.arena);
            
            % Update navigator's grid position to match robot
            obj.position = robot.gridPos;
            obj.pathHistory(end+1, :) = obj.position;
            
            fprintf("[NAV] Moved to cell (%d,%d), Robot grid (%d,%d)\n", ...
                nextCell, robot.gridPos);
        end
        
        function target = chooseBestR2(obj, remainingTime)
            % Choose best R2 target using hybrid cost
            target = NavigationPlanner.chooseBestR2(obj.position, obj.belief, ...
                obj.beliefConfidence, obj.carry, obj.arena.getHeightMap(), ...
                obj.config.exitCells, remainingTime, obj.config);
        end
        
        function target = findBestExploreCell(obj)
            target = NavigationPlanner.findBestExploreCell(obj.position, ...
                obj.belief, obj.beliefConfidence, obj.arena.getHeightMap(), ...
                obj.visitedCells, obj.config);
        end
        
        function [target, blockedR1] = planExitRoute(obj)
            [target, blockedR1] = NavigationPlanner.planExitRoute(obj.position, ...
                obj.belief, obj.beliefConfidence, obj.config.exitCells, ...
                obj.arena, obj.config);
        end
        
        function reset(obj, arena)
            % Reset navigator state for new mission
            obj.mode = "COLLECT";
            obj.carry = 0;
            
            initialBlock = arena.getInitialBlock();
            obj.position = [initialBlock.row, initialBlock.col];
            
            obj.belief(:) = "UNSEEN";
            obj.beliefConfidence(:) = 0;
            
            obj.visitedCells(:) = false;
            obj.visitedCells(obj.position(1), obj.position(2)) = true;
            
            obj.pathHistory = obj.position;
            obj.gameStartTime = tic;
            obj.emergencyExitTriggered = false;
            
            obj.r1WaitStart = containers.Map;
            obj.r1WaitAttempts = containers.Map;
            
            obj.lastGridPos = obj.position;
            obj.stuckCounter = 0;
            obj.replanCounter = 0;
            
            obj.collectedR2Positions = [];
            obj.pathFailureCount = 0;
            obj.missionComplete = false;
            
            fprintf('[NAV] Navigator reset for new mission\n');
        end
        
        function updateBeliefFromDetections(obj, vision, arena)
            % Update belief map from all detected KFS (optimization)
            % This allows navigator to know about detected KFS even when not in view
            
            if isempty(vision.detectedKfs)
                return;
            end
            
            detectedIds = cell2mat(vision.detectedKfs.keys);
            
            for i = 1:length(detectedIds)
                kfsId = detectedIds(i);
                detection = vision.detectedKfs(kfsId);
                
                % Find block grid position
                block = arena.blocks(kfsId);
                row = block.row;
                col = block.col;
                
                % DON'T overwrite if already collected (belief = EMPTY)
                % This prevents re-detecting collected items
                if obj.belief(row, col) == "EMPTY"
                    continue;
                end
                
                % Update belief with detected type and confidence
                obj.belief(row, col) = detection.type;
                obj.beliefConfidence(row, col) = max(obj.beliefConfidence(row, col), ...
                    detection.confidence);
            end
        end
    end
end
