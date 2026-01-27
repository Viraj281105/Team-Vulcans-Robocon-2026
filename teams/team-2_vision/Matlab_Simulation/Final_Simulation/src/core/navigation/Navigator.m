classdef Navigator < handle
    % NAVIGATOR Time-aware hybrid A* path planner with vision-before-move
    
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
        cellConfirmed   % NEW: Track vision-confirmed cells
        
        % Path planning
        pathHistory
        pendingTarget   % NEW: Cell waiting for vision confirmation
        confirmWaitStart % NEW: When we started waiting for confirmation
        
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
        MAX_REPLAN_ATTEMPTS = 25
        MIN_PLAN_CONF = 0.4
        VISION_CONFIRM_TIMEOUT = 3.0  % NEW: Max seconds to wait for vision
    end
    
    methods
        function obj = Navigator(config, arena)
            obj.config = config;
            obj.arena = arena;
            obj.mode = "COLLECT";
            obj.carry = 0;
            
            initialBlock = arena.getInitialBlock();
            
            % FIXED: Handle ground start position (row=0)
            if initialBlock.row == 0
                % Robot starts on ground - use first row, middle column as initial grid target
                obj.position = [1, initialBlock.col];
                fprintf('[NAV] Robot starting on GROUND - initial grid target: (%d,%d)\n', ...
                    obj.position);
            else
                % Robot starts on a block
                obj.position = [initialBlock.row, initialBlock.col];
            end
            
            obj.belief = strings(config.FOREST_ROWS, config.FOREST_COLS);
            obj.belief(:) = "UNSEEN";
            obj.beliefConfidence = zeros(config.FOREST_ROWS, config.FOREST_COLS);
            obj.cellConfirmed = false(config.FOREST_ROWS, config.FOREST_COLS);  % NEW
            
            obj.visitedCells = false(config.FOREST_ROWS, config.FOREST_COLS);
            
            % FIXED: Only mark as visited if starting inside forest (row >= 1)
            if obj.isInForestGrid(obj.position)
                obj.visitedCells(obj.position(1), obj.position(2)) = true;
                obj.cellConfirmed(obj.position(1), obj.position(2)) = true;  % Starting cell confirmed
            end
            
            obj.pathHistory = obj.position;
            obj.pendingTarget = [];  % NEW
            obj.confirmWaitStart = [];  % NEW
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
        
        function inGrid = isInForestGrid(obj, pos)
            % NEW: Helper to check if position is within forest grid bounds
            cfg = obj.config;
            inGrid = pos(1) >= 1 && pos(1) <= cfg.FOREST_ROWS && ...
                     pos(2) >= 1 && pos(2) <= cfg.FOREST_COLS;
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
                obj.carry = max(obj.carry, 1);  % Don't exit empty-handed
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
            cameraYaw = robot.cameraYaw;
            cameraPitch = robot.cameraPitch;
            
            [obj.belief, obj.beliefConfidence] = vision.updateGridBelief(...
                cameraPos, cameraYaw, cameraPitch, obj.belief, obj.beliefConfidence);
            
            % FIXED: Only mark visited if inside forest grid
            if obj.isInForestGrid(obj.position)
                obj.visitedCells(obj.position(1), obj.position(2)) = true;
            end
            
            % Stuck detection
            obj.detectStuck();
            
            % CRITICAL: Check for pickup BEFORE planning
            if obj.tryPickKFS(arena)
                % Check if we've reached capacity after pickup
                if obj.carry >= cfg.CAPACITY && obj.mode ~= "EXIT"
                    fprintf("[NAV] Capacity full (%d/%d) → EXIT mode\n", obj.carry, cfg.CAPACITY);
                    obj.mode = "EXIT";
                end
                return;  % Done for this step
            end
            
            % Check if at exit (only complete if full capacity or emergency)
            if obj.checkExitReached()
                return;
            end
            
            % R1 waiting logic
            if obj.mode == "WAIT_FOR_EXIT"
                obj.handleR1Waiting(arena);
                return;
            end
            
            % Plan and execute move
            obj.planAndMoveWithVision(robot, arena, vision, remainingTime);
            
            % CRITICAL: Check for pickup again AFTER movement
            if obj.tryPickKFS(arena)
                fprintf("[NAV] Picked up R2 after movement!\n");
                % Check if we've reached capacity
                if obj.carry >= cfg.CAPACITY && obj.mode ~= "EXIT"
                    fprintf("[NAV] Capacity full (%d/%d) → EXIT mode\n", obj.carry, cfg.CAPACITY);
                    obj.mode = "EXIT";
                end
                return;
            end
        end


        
        function planAndMoveWithVision(obj, robot, arena, vision, remainingTime)
            % NEW: Plan and move WITH vision confirmation requirement
            
            % CRITICAL FIX: If on ground, MUST enter forest first
            if robot.position(3) <= 10 && ~arena.isInForestBounds(robot.position(1:2))
                fprintf("[NAV] On GROUND - must enter forest before exploring\n");
                
                % Force target to row 1 (any unseen cell in first row)
                cfg = obj.config;
                entryTarget = [];
                
                % Try to find an unseen cell in row 1
                for c = 1:cfg.FOREST_COLS
                    if obj.belief(1, c) == "UNSEEN" || obj.belief(1, c) == "UNKNOWN"
                        entryTarget = [1, c];
                        break;
                    end
                end
                
                % If all row 1 cells are believed known, just pick middle column
                if isempty(entryTarget)
                    entryTarget = [1, 2];  % Middle column of row 1
                end
                
                fprintf("[NAV] Forcing entry at row 1, col %d\n", entryTarget(2));
                obj.executeMove(entryTarget, robot, arena);
                return;
            end
            
            % Check if we're waiting for vision confirmation
            if ~isempty(obj.pendingTarget)
                r = obj.pendingTarget(1);
                c = obj.pendingTarget(2);
                
                % Check confirmation timeout
                waitTime = toc(obj.confirmWaitStart);
                if waitTime > obj.VISION_CONFIRM_TIMEOUT
                    fprintf("[NAV] Vision confirm timeout at (%d,%d) after %.1fs - SKIPPING\n", r, c, waitTime);
                    obj.cellConfirmed(r, c) = true;
                    obj.belief(r, c) = "EMPTY";
                    obj.beliefConfidence(r, c) = 0.5;
                    obj.pendingTarget = [];
                    return;
                end
                
                % Auto-focus camera on pending cell
                obj.focusCameraOnCell(r, c, robot, arena);
                
                % Check if vision confirms this cell now
                if obj.beliefConfidence(r, c) >= obj.config.VISION_CONFIDENCE_THRESHOLD
                    obj.cellConfirmed(r, c) = true;
                    obj.pendingTarget = [];
                    fprintf("[NAV] Cell (%d,%d) confirmed as %s (conf=%.2f)\n", ...
                        r, c, obj.belief(r, c), obj.beliefConfidence(r, c));
                else
                    fprintf("[NAV] Waiting for vision at (%d,%d) | conf=%.2f | waited=%.1fs\n", ...
                        r, c, obj.beliefConfidence(r, c), waitTime);
                    return;
                end
            end
            
            % Regular planning logic
            target = [];
            
            if obj.mode == "COLLECT"
                target = obj.chooseBestR2(remainingTime);
                
                if isempty(target)
                    r2Count = sum(obj.belief == "R2", 'all');
                    if r2Count > 0
                        fprintf("[NAV DEBUG] %d R2(s) in belief but none selected\n", r2Count);
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
            
            % Execute move (check if target needs confirmation)
            if ~isempty(target)
                r = target(1);
                c = target(2);
                
                % NEW: Skip vision confirmation for adjacent cells in same row
                isAdjacentSameRow = (abs(r - obj.position(1)) == 0) && ...
                                    (abs(c - obj.position(2)) == 1);
                
                % Check if we need vision confirmation before moving
                if ~obj.cellConfirmed(r, c) && obj.mode == "COLLECT" && ~isAdjacentSameRow
                    % Start waiting for vision confirmation
                    obj.pendingTarget = target;
                    obj.confirmWaitStart = tic;
                    obj.focusCameraOnCell(r, c, robot, arena);
                    fprintf("[NAV] Requesting vision confirmation for (%d,%d)\n", r, c);
                    return;
                else
                    % Already confirmed OR in EXIT mode OR adjacent same-row cell
                    if isAdjacentSameRow
                        fprintf("[NAV] Moving to adjacent cell (%d,%d) - skipping vision confirm\n", r, c);
                    end
                    obj.executeMove(target, robot, arena);
                end
            end
        end

        
        function focusCameraOnCell(obj, r, c, robot, arena)
            % NEW: Auto-focus camera on target grid cell
            cfg = obj.config;
            
            % Find target block
            targetBlock = [];
            for k = 1:numel(arena.blocks)
                b = arena.blocks(k);
                if b.row == r && b.col == c
                    targetBlock = b;
                    break;
                end
            end
            
            if isempty(targetBlock)
                return;
            end
            
            % Calculate 3D target position (top center of KFS)
            target_x = targetBlock.x;
            target_y = targetBlock.y;
            target_z = targetBlock.h;  % Height of block top surface
            
            % Robot camera position
            robotPos = robot.position;
            cameraPos = robotPos + [0 0 cfg.CAMERA_HEIGHT_OFFSET];
            
            % Calculate required camera angles
            dx = target_x - cameraPos(1);
            dy = target_y - cameraPos(2);
            dz = target_z - cameraPos(3);
            
            % Yaw (horizontal rotation) - MATLAB convention
            target_yaw = atan2d(dy, dx) + 90;  % Adjust for MATLAB coordinate system
            
            % Pitch (vertical tilt) - negative = looking down
            horizontal_dist = sqrt(dx^2 + dy^2);
            target_pitch = -atan2d(dz, horizontal_dist);
            
            % Smooth interpolation for realistic camera movement
            CAMERA_TURN_SPEED = 0.3;  % 0=instant, 1=no change
            robot.cameraYaw = robot.cameraYaw * (1-CAMERA_TURN_SPEED) + target_yaw * CAMERA_TURN_SPEED;
            robot.cameraPitch = robot.cameraPitch * (1-CAMERA_TURN_SPEED) + target_pitch * CAMERA_TURN_SPEED;
            
            % Clamp pitch to valid range
            robot.cameraPitch = max(min(robot.cameraPitch, 45), -45);
            
            fprintf("[CAMERA] Auto-focus: (%d,%d) | Yaw: %.1f° | Pitch: %.1f°\n", ...
                r, c, robot.cameraYaw, robot.cameraPitch);
        end
        
        function detectStuck(obj)
            if norm(obj.position - obj.lastGridPos) < obj.POSITION_TOLERANCE
                obj.stuckCounter = obj.stuckCounter + 1;
                if obj.stuckCounter >= obj.MAX_STUCK_COUNT
                    fprintf("[NAV] Stuck detected, re-evaluating...\n");
                    obj.stuckCounter = 0;
                    obj.replanCounter = obj.replanCounter + 1;
                    
                    % FIXED: Only force EXIT if capacity is full OR time is critical
                    if obj.replanCounter >= obj.MAX_REPLAN_ATTEMPTS
                        elapsedTime = toc(obj.gameStartTime);
                        remainingTime = obj.config.TIME_LIMIT - elapsedTime;
                        
                        % Only exit if we have items AND (capacity full OR low time OR emergency)
                        if obj.carry >= obj.config.CAPACITY
                            fprintf("[NAV] Max replans + full capacity → forcing EXIT\n");
                            obj.mode = "EXIT";
                            obj.emergencyExitTriggered = true;
                        elseif remainingTime < 30 && obj.carry > 0
                            fprintf("[NAV] Max replans + low time (%.1fs) → forcing EXIT\n", remainingTime);
                            obj.mode = "EXIT";
                            obj.emergencyExitTriggered = true;
                        else
                            % Reset replan counter and continue collecting
                            fprintf("[NAV] Max replans but only %d/%d items → RESET counter, continue collecting\n", ...
                                obj.carry, obj.config.CAPACITY);
                            obj.replanCounter = 0;  % Reset and keep exploring
                        end
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
            
            if ~obj.isInForestGrid(obj.position)
                return;
            end
            
            r = obj.position(1);
            c = obj.position(2);
            cellBelief = obj.belief(r, c);
            cellConf = obj.beliefConfidence(r, c);
            
            % ALWAYS show debug when on a believed R2 cell
            if cellBelief == "R2"
                fprintf("[NAV DEBUG] At (%d,%d): belief=R2, conf=%.2f, threshold=%.2f\n", ...
                    r, c, cellConf, cfg.VISION_CONFIDENCE_THRESHOLD * 0.85);
            end
            
            requiredConfidence = cfg.VISION_CONFIDENCE_THRESHOLD * 0.85;
            
            if cellBelief == "R2" && cellConf >= requiredConfidence
                obj.carry = obj.carry + 1;
                arena.setTruthAt(r, c, "EMPTY");
                obj.belief(r, c) = "EMPTY";
                obj.collectedR2Positions = [obj.collectedR2Positions; obj.position];
                
                fprintf("[NAV] ✓ PICK at (%d,%d), carry=%d/%d (conf=%.2f)\n", ...
                    r, c, obj.carry, cfg.CAPACITY, cellConf);
                
                picked = true;
            elseif cellBelief == "R2"
                fprintf("[NAV WARNING] R2 at (%d,%d) but conf too low: %.2f < %.2f (MISSED!)\n", ...
                    r, c, cellConf, requiredConfidence);
            end
        end


        
        function reached = checkExitReached(obj)
            reached = false;
            if ismember(obj.position, obj.config.exitCells, 'rows')
                % FIXED: Only complete mission if capacity is FULL or in emergency exit
                if obj.carry >= obj.config.CAPACITY
                    fprintf("[NAV] ✓ MISSION COMPLETE! Reached exit at (%d,%d) with %d items (FULL CAPACITY)\n", ...
                        obj.position, obj.carry);
                    obj.missionComplete = true;
                    reached = true;
                elseif obj.emergencyExitTriggered
                    fprintf("[NAV] ✓ EMERGENCY EXIT! Reached exit at (%d,%d) with %d items (TIME LIMIT)\n", ...
                        obj.position, obj.carry);
                    obj.missionComplete = true;
                    reached = true;
                else
                    fprintf("[NAV] At exit cell (%d,%d) but only carrying %d/%d items - continue collecting\n", ...
                        obj.position, obj.carry, obj.config.CAPACITY);
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
        
        function executeMove(obj, target, robot, arena)
            cfg = obj.config;
            
            % Check if already at target
            if isequal(obj.position, target)
                fprintf("[NAV] Already at target (%d,%d) - no movement needed\n", target);
                return;
            end
            
            % CRITICAL FIX: Handle ground-to-forest entry
            if robot.position(3) <= 10  % On ground level
                if ~arena.isInForestBounds(robot.position(1:2))
                    fprintf("[NAV] On GROUND outside forest - entering at row 1, col %d\n", target(2));
                    
                    targetBlock = [];
                    targetRow = min(target(1), 1);
                    
                    for k = 1:numel(arena.blocks)
                        b = arena.blocks(k);
                        if b.row == targetRow && b.col == target(2)
                            targetBlock = b;
                            break;
                        end
                    end
                    
                    if ~isempty(targetBlock)
                        robot.moveTo([targetBlock.x, targetBlock.y], arena);
                        obj.position = robot.gridPos;
                        obj.pathHistory(end+1, :) = obj.position;
                        
                        % CRITICAL FIX: Force high-confidence belief update when entering
                        if obj.belief(obj.position(1), obj.position(2)) == "R2"
                            obj.beliefConfidence(obj.position(1), obj.position(2)) = 0.9;
                            fprintf("[NAV] ✓ Entered forest at grid (%d,%d) - boosted R2 confidence to 0.9\n", ...
                                obj.position);
                        else
                            fprintf("[NAV] ✓ Entered forest at grid (%d,%d), 3D pos (%.0f, %.0f)\n", ...
                                obj.position, robot.position(1), robot.position(2));
                        end
                        return;
                    else
                        fprintf("[NAV ERROR] Could not find entry block for row %d, col %d\n", ...
                            targetRow, target(2));
                        return;
                    end
                end
            end
            
            % NORMAL CASE: Already in forest, use A* pathfinding
            pathPlanner = PathPlanner(obj.config, obj.arena);
            path = pathPlanner.astarWithTerrain(obj.position, target, ...
                obj.belief, obj.beliefConfidence, obj.mode);
            
            if isempty(path) || size(path, 1) < 2
                fprintf("[NAV] No path to (%d,%d)\n", target);
                return;
            end
            
            nextCell = path(2, :);
            
            fprintf("[NAV DEBUG] Current: (%d,%d), Target: (%d,%d), Next: (%d,%d), PathLen: %d\n", ...
                obj.position(1), obj.position(2), target(1), target(2), ...
                nextCell(1), nextCell(2), size(path,1));
            
            if abs(nextCell(1) - obj.position(1)) + abs(nextCell(2) - obj.position(2)) ~= 1
                fprintf("[NAV] Non-adjacent step, skipping\n");
                return;
            end
            
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
            
            robot.moveTo([bNext.x, bNext.y], obj.arena);
            obj.position = robot.gridPos;
            obj.pathHistory(end+1, :) = obj.position;
            
            % CRITICAL FIX: Boost confidence when moving onto R2
            if obj.belief(obj.position(1), obj.position(2)) == "R2"
                obj.beliefConfidence(obj.position(1), obj.position(2)) = 0.9;
                fprintf("[NAV] Moved to R2 cell (%d,%d) - boosted confidence to 0.9\n", obj.position);
            else
                fprintf("[NAV] Moved to cell (%d,%d), Robot grid (%d,%d)\n", ...
                    nextCell, robot.gridPos);
            end
        end


        
        function target = chooseBestR2(obj, remainingTime)
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
            obj.mode = "COLLECT";
            obj.carry = 0;
            
            initialBlock = arena.getInitialBlock();
            
            % FIXED: Handle ground start position
            if initialBlock.row == 0
                obj.position = [1, initialBlock.col];
            else
                obj.position = [initialBlock.row, initialBlock.col];
            end
            
            obj.belief(:) = "UNSEEN";
            obj.beliefConfidence(:) = 0;
            obj.cellConfirmed(:) = false;  % NEW: Reset confirmations
            
            obj.visitedCells(:) = false;
            
            % FIXED: Only mark as visited if inside forest grid
            if obj.isInForestGrid(obj.position)
                obj.visitedCells(obj.position(1), obj.position(2)) = true;
                obj.cellConfirmed(obj.position(1), obj.position(2)) = true;
            end
            
            obj.pathHistory = obj.position;
            obj.pendingTarget = [];  % NEW
            obj.confirmWaitStart = [];  % NEW
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
            if isempty(vision.detectedKfs)
                return;
            end
            
            detectedIds = cell2mat(vision.detectedKfs.keys);
            
            for i = 1:length(detectedIds)
                kfsId = detectedIds(i);
                detection = vision.detectedKfs(kfsId);
                
                block = arena.blocks(kfsId);
                row = block.row;
                col = block.col;
                
                if obj.belief(row, col) == "EMPTY"
                    continue;
                end
                
                obj.belief(row, col) = detection.type;
                obj.beliefConfidence(row, col) = max(obj.beliefConfidence(row, col), ...
                    detection.confidence);
            end
        end
    end
end
