classdef Navigator < handle
    % ===================================================================
    % NAVIGATOR
    % -------------------------------------------------------------------
    % Time-aware hybrid A* autonomous navigation controller with:
    %   - Vision-driven belief updates
    %   - Confidence-aware planning
    %   - Time-pressure emergency handling
    %   - Stuck detection and recovery
    %   - Vision-before-move safety gating
    %   - Exit logic and obstacle waiting
    %
    % Architectural role:
    %   - Acts as the mission-level executive controller.
    %   - Orchestrates perception, planning, decision-making, and motion.
    %   - Maintains internal state machines and safety logic.
    %
    % Think of this as the robot's "autonomous brainstem."
    % ===================================================================
    
    properties
        % ------------------ Core References ------------------------------
        config
        arena
        
        % ------------------ Navigation State -----------------------------
        mode                    % "COLLECT", "EXIT", "WAIT_FOR_EXIT"
        carry                   % Number of R2 items currently carried
        position                % Current grid position [row, col]
        
        % ------------------ Belief State --------------------------------
        belief                  % Grid belief map (string labels)
        beliefConfidence         % Confidence per grid cell
        visitedCells             % Boolean visited flags
        cellConfirmed           % Vision-confirmed cells (safety gating)
        
        % ------------------ Path Planning -------------------------------
        pathHistory              % Grid trajectory history
        pendingTarget            % Cell awaiting vision confirmation
        confirmWaitStart         % Timer for confirmation waiting
        
        % ------------------ Timing -------------------------------------
        gameStartTime
        emergencyExitTriggered
        
        % ------------------ R1 Waiting ---------------------------------
        r1WaitStart
        r1WaitAttempts
        
        % ------------------ Stuck Detection ------------------------------
        lastGridPos
        stuckCounter
        replanCounter
        
        % ------------------ Collection Tracking -------------------------
        collectedR2Positions
        pathFailureCount
        
        % ------------------ Mission Completion --------------------------
        missionComplete
    end
    
    properties (Constant)
        POSITION_TOLERANCE      = 0.1
        MAX_STUCK_COUNT         = 3
        MAX_REPLAN_ATTEMPTS     = 25
        MIN_PLAN_CONF           = 0.4
        VISION_CONFIRM_TIMEOUT  = 3.0     % Max seconds to wait for vision
    end
    
    methods
        
        % ===================================================================
        % Constructor
        % -------------------------------------------------------------------
        % Initializes navigation state, belief grid, counters, and timers.
        % ===================================================================
        function obj = Navigator(config, arena)
            
            obj.config = config;
            obj.arena  = arena;
            
            % Initial mission mode
            obj.mode  = "COLLECT";
            obj.carry = 0;
            
            % Determine initial grid position
            initialBlock = arena.getInitialBlock();
            
            % Handle ground start condition explicitly
            if initialBlock.row == 0
                obj.position = [1, initialBlock.col];
                fprintf('[NAV] Robot starting on GROUND - initial grid target: (%d,%d)\n', ...
                    obj.position);
            else
                obj.position = [initialBlock.row, initialBlock.col];
            end
            
            % Initialize belief maps
            obj.belief = strings(config.FOREST_ROWS, config.FOREST_COLS);
            obj.belief(:) = "UNSEEN";
            
            obj.beliefConfidence = zeros(config.FOREST_ROWS, config.FOREST_COLS);
            obj.cellConfirmed    = false(config.FOREST_ROWS, config.FOREST_COLS);
            obj.visitedCells     = false(config.FOREST_ROWS, config.FOREST_COLS);
            
            % Mark starting cell as visited and confirmed (if valid)
            if obj.isInForestGrid(obj.position)
                obj.visitedCells(obj.position(1), obj.position(2)) = true;
                obj.cellConfirmed(obj.position(1), obj.position(2)) = true;
            end
            
            % Initialize path history
            obj.pathHistory       = obj.position;
            obj.pendingTarget     = [];
            obj.confirmWaitStart  = [];
            
            % Timing
            obj.gameStartTime = tic;
            obj.emergencyExitTriggered = false;
            
            % R1 waiting maps
            obj.r1WaitStart    = containers.Map;
            obj.r1WaitAttempts = containers.Map;
            
            % Stuck detection state
            obj.lastGridPos   = obj.position;
            obj.stuckCounter  = 0;
            obj.replanCounter = 0;
            
            % Collection bookkeeping
            obj.collectedR2Positions = [];
            obj.pathFailureCount     = 0;
            obj.missionComplete      = false;
        end
        
        
        % ===================================================================
        % Forest Grid Bounds Check
        % -------------------------------------------------------------------
        % Safely validates grid coordinates.
        % ===================================================================
        function inGrid = isInForestGrid(obj, pos)
            cfg = obj.config;
            inGrid = ...
                pos(1) >= 1 && pos(1) <= cfg.FOREST_ROWS && ...
                pos(2) >= 1 && pos(2) <= cfg.FOREST_COLS;
        end
        
        
        % ===================================================================
        % Main Autonomous Step
        % -------------------------------------------------------------------
        % Executes one closed-loop autonomous iteration.
        % ===================================================================
        function step(obj, robot, arena, vision, detections)
            
            cfg = obj.config;
            
            % Sync grid position from robot state
            obj.position = robot.gridPos;
            
            % ------------------ Time Monitoring ---------------------------
            elapsedTime   = toc(obj.gameStartTime);
            remainingTime = cfg.TIME_LIMIT - elapsedTime;
            
            % Emergency exit trigger under low time
            if remainingTime < cfg.TIME_BUFFER && ...
               ~obj.emergencyExitTriggered
                
                fprintf("[NAV] EMERGENCY: Only %.1fs left → EXIT mode\n", remainingTime);
                obj.mode = "EXIT";
                obj.emergencyExitTriggered = true;
                obj.carry = max(obj.carry, 1);
            end
            
            % Hard timeout protection
            if remainingTime <= 0
                fprintf("[NAV] TIMEOUT, stopping autonomous\n");
                return;
            end
            
            % ------------------ Belief Updates -----------------------------
            % Update belief from confirmed detections
            obj.updateBeliefFromDetections(vision, arena);
            
            % Update belief from current camera visibility
            robotState  = robot.getState();
            cameraPos   = robotState.position + [0 0 cfg.CAMERA_HEIGHT_OFFSET];
            cameraYaw   = robot.cameraYaw;
            cameraPitch = robot.cameraPitch;
            
            [obj.belief, obj.beliefConfidence] = ...
                vision.updateGridBelief( ...
                    cameraPos, cameraYaw, cameraPitch, ...
                    obj.belief, obj.beliefConfidence);
            
            % Mark visited cell safely
            if obj.isInForestGrid(obj.position)
                obj.visitedCells(obj.position(1), obj.position(2)) = true;
            end
            
            % ------------------ Stuck Detection ----------------------------
            obj.detectStuck();
            
            % ------------------ Pickup Check (Pre-move) --------------------
            if obj.tryPickKFS(arena)
                
                if obj.carry >= cfg.CAPACITY && obj.mode ~= "EXIT"
                    fprintf("[NAV] Capacity full (%d/%d) → EXIT mode\n", ...
                        obj.carry, cfg.CAPACITY);
                    obj.mode = "EXIT";
                end
                return;
            end
            
            % ------------------ Exit Check --------------------------------
            if obj.checkExitReached()
                return;
            end
            
            % ------------------ R1 Waiting Mode ----------------------------
            if obj.mode == "WAIT_FOR_EXIT"
                obj.handleR1Waiting(arena);
                return;
            end
            
            % ------------------ Planning + Movement ------------------------
            obj.planAndMoveWithVision(robot, arena, vision, remainingTime);
            
            % ------------------ Pickup Check (Post-move) -------------------
            if obj.tryPickKFS(arena)
                fprintf("[NAV] Picked up R2 after movement!\n");
                
                if obj.carry >= cfg.CAPACITY && obj.mode ~= "EXIT"
                    fprintf("[NAV] Capacity full (%d/%d) → EXIT mode\n", ...
                        obj.carry, cfg.CAPACITY);
                    obj.mode = "EXIT";
                end
                return;
            end
        end
        
        % ===================================================================
        % Vision-Gated Planning and Movement
        % -------------------------------------------------------------------
        % Enforces camera confirmation before moving into unverified cells.
        % ===================================================================
        function planAndMoveWithVision(obj, robot, arena, vision, remainingTime)
            
            % ------------------ Ground Entry Enforcement ------------------
            if robot.position(3) <= 10 && ...
               ~arena.isInForestBounds(robot.position(1:2))
                
                fprintf("[NAV] On GROUND - must enter forest before exploring\n");
                
                cfg = obj.config;
                entryTarget = [];
                
                % Prefer unseen cells in first row
                for c = 1:cfg.FOREST_COLS
                    if obj.belief(1, c) == "UNSEEN" || ...
                       obj.belief(1, c) == "UNKNOWN"
                        entryTarget = [1, c];
                        break;
                    end
                end
                
                if isempty(entryTarget)
                    entryTarget = [1, 2];
                end
                
                fprintf("[NAV] Forcing entry at row 1, col %d\n", entryTarget(2));
                obj.executeMove(entryTarget, robot, arena);
                return;
            end
            
            % ------------------ Vision Confirmation Wait ------------------
            if ~isempty(obj.pendingTarget)
                
                r = obj.pendingTarget(1);
                c = obj.pendingTarget(2);
                
                waitTime = toc(obj.confirmWaitStart);
                
                % Timeout handling
                if waitTime > obj.VISION_CONFIRM_TIMEOUT
                    fprintf("[NAV] Vision confirm timeout at (%d,%d) after %.1fs - SKIPPING\n", ...
                        r, c, waitTime);
                    
                    obj.cellConfirmed(r, c) = true;
                    obj.belief(r, c) = "EMPTY";
                    obj.beliefConfidence(r, c) = 0.5;
                    obj.pendingTarget = [];
                    return;
                end
                
                % Camera auto-focus for confirmation
                obj.focusCameraOnCell(r, c, robot, arena);
                
                % Check confirmation threshold
                if obj.beliefConfidence(r, c) >= ...
                        obj.config.VISION_CONFIDENCE_THRESHOLD
                    
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
            
            % ------------------ Target Selection ---------------------------
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
            
            % ------------------ Execution with Vision Gating ---------------
            if ~isempty(target)
                
                r = target(1);
                c = target(2);
                
                % Adjacent same-row cells skip vision confirmation
                isAdjacentSameRow = ...
                    (abs(r - obj.position(1)) == 0) && ...
                    (abs(c - obj.position(2)) == 1);
                
                % Require confirmation for non-adjacent exploration moves
                if ~obj.cellConfirmed(r, c) && ...
                   obj.mode == "COLLECT" && ...
                   ~isAdjacentSameRow
                    
                    obj.pendingTarget     = target;
                    obj.confirmWaitStart  = tic;
                    
                    obj.focusCameraOnCell(r, c, robot, arena);
                    
                    fprintf("[NAV] Requesting vision confirmation for (%d,%d)\n", r, c);
                    return;
                    
                else
                    if isAdjacentSameRow
                        fprintf("[NAV] Moving to adjacent cell (%d,%d) - skipping vision confirm\n", r, c);
                    end
                    
                    obj.executeMove(target, robot, arena);
                end
            end
        end
        
        
        % ===================================================================
        % Camera Auto-Focus Utility
        % -------------------------------------------------------------------
        % Aligns camera toward target grid cell for visual confirmation.
        % ===================================================================
        function focusCameraOnCell(obj, r, c, robot, arena)
            
            cfg = obj.config;
            
            % Locate block matching grid coordinates
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
            
            % Target 3D coordinates
            target_x = targetBlock.x;
            target_y = targetBlock.y;
            target_z = targetBlock.h;
            
            % Camera position
            robotPos  = robot.position;
            cameraPos = robotPos + [0 0 cfg.CAMERA_HEIGHT_OFFSET];
            
            % Vector to target
            dx = target_x - cameraPos(1);
            dy = target_y - cameraPos(2);
            dz = target_z - cameraPos(3);
            
            % Compute yaw (horizontal)
            target_yaw = atan2d(dy, dx) + 90;
            
            % Compute pitch (vertical)
            horizontal_dist = sqrt(dx^2 + dy^2);
            target_pitch = -atan2d(dz, horizontal_dist);
            
            % Smooth camera interpolation
            CAMERA_TURN_SPEED = 0.3;
            
            robot.cameraYaw   = ...
                robot.cameraYaw   * (1-CAMERA_TURN_SPEED) + ...
                target_yaw        * CAMERA_TURN_SPEED;
            
            robot.cameraPitch = ...
                robot.cameraPitch * (1-CAMERA_TURN_SPEED) + ...
                target_pitch      * CAMERA_TURN_SPEED;
            
            % Clamp pitch to physical limits
            robot.cameraPitch = ...
                max(min(robot.cameraPitch, 45), -45);
            
            fprintf("[CAMERA] Auto-focus: (%d,%d) | Yaw: %.1f° | Pitch: %.1f°\n", ...
                r, c, robot.cameraYaw, robot.cameraPitch);
        end
        
        
        % ===================================================================
        % Stuck Detection and Recovery
        % -------------------------------------------------------------------
        % Detects lack of movement and escalates recovery actions.
        % ===================================================================
        function detectStuck(obj)
            
            if norm(obj.position - obj.lastGridPos) < ...
                    obj.POSITION_TOLERANCE
                
                obj.stuckCounter = obj.stuckCounter + 1;
                
                if obj.stuckCounter >= obj.MAX_STUCK_COUNT
                    
                    fprintf("[NAV] Stuck detected, re-evaluating...\n");
                    obj.stuckCounter = 0;
                    obj.replanCounter = obj.replanCounter + 1;
                    
                    if obj.replanCounter >= obj.MAX_REPLAN_ATTEMPTS
                        
                        elapsedTime   = toc(obj.gameStartTime);
                        remainingTime = obj.config.TIME_LIMIT - elapsedTime;
                        
                        if obj.carry >= obj.config.CAPACITY
                            fprintf("[NAV] Max replans + full capacity → forcing EXIT\n");
                            obj.mode = "EXIT";
                            obj.emergencyExitTriggered = true;
                            
                        elseif remainingTime < 30 && obj.carry > 0
                            fprintf("[NAV] Max replans + low time (%.1fs) → forcing EXIT\n", ...
                                remainingTime);
                            obj.mode = "EXIT";
                            obj.emergencyExitTriggered = true;
                            
                        else
                            fprintf("[NAV] Max replans but only %d/%d items → RESET counter, continue collecting\n", ...
                                obj.carry, obj.config.CAPACITY);
                            obj.replanCounter = 0;
                        end
                    end
                end
            else
                obj.stuckCounter = 0;
                obj.lastGridPos  = obj.position;
            end
        end
        
        
        % ===================================================================
        % Pickup Logic
        % -------------------------------------------------------------------
        % Attempts to collect R2 item at current grid cell.
        % ===================================================================
        function picked = tryPickKFS(obj, arena)
            
            picked = false;
            cfg    = obj.config;
            
            if ~obj.isInForestGrid(obj.position)
                return;
            end
            
            r = obj.position(1);
            c = obj.position(2);
            
            cellBelief = obj.belief(r, c);
            cellConf   = obj.beliefConfidence(r, c);
            
            % Debug visibility
            if cellBelief == "R2"
                fprintf("[NAV DEBUG] At (%d,%d): belief=R2, conf=%.2f, threshold=%.2f\n", ...
                    r, c, cellConf, cfg.VISION_CONFIDENCE_THRESHOLD * 0.85);
            end
            
            requiredConfidence = cfg.VISION_CONFIDENCE_THRESHOLD * 0.85;
            
            if cellBelief == "R2" && ...
               cellConf >= requiredConfidence
                
                obj.carry = obj.carry + 1;
                
                arena.setTruthAt(r, c, "EMPTY");
                obj.belief(r, c) = "EMPTY";
                
                obj.collectedR2Positions = ...
                    [obj.collectedR2Positions; obj.position];
                
                fprintf("[NAV] ✓ PICK at (%d,%d), carry=%d/%d (conf=%.2f)\n", ...
                    r, c, obj.carry, cfg.CAPACITY, cellConf);
                
                picked = true;
                
            elseif cellBelief == "R2"
                
                fprintf("[NAV WARNING] R2 at (%d,%d) but conf too low: %.2f < %.2f (MISSED!)\n", ...
                    r, c, cellConf, requiredConfidence);
            end
        end
        
        
        % ===================================================================
        % Exit Completion Check
        % -------------------------------------------------------------------
        % Determines whether mission termination conditions are satisfied.
        % ===================================================================
        function reached = checkExitReached(obj)
            
            reached = false;
            
            if ismember(obj.position, obj.config.exitCells, 'rows')
                
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
        
        
        % ===================================================================
        % R1 Waiting Logic
        % -------------------------------------------------------------------
        % Waits probabilistically for R1 obstacle clearance.
        % ===================================================================
        function handleR1Waiting(obj, arena)
            
            cfg    = obj.config;
            keysR1 = obj.r1WaitStart.keys;
            
            if isempty(keysR1)
                obj.mode = "EXIT";
                return;
            end
            
            kR1       = keysR1{1};
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
                fprintf("[NAV] WAITING FOR R1 (%.1fs / %ds)\n", ...
                    elapsedR1, cfg.WAIT_TIME);
            end
        end
        
        
        % ===================================================================
        % Execute Movement Toward Target
        % -------------------------------------------------------------------
        % Handles both ground entry and in-forest path execution.
        % ===================================================================
        function executeMove(obj, target, robot, arena)
            
            cfg = obj.config;
            
            % Already at target
            if isequal(obj.position, target)
                fprintf("[NAV] Already at target (%d,%d) - no movement needed\n", target);
                return;
            end
            
            % ------------------ Ground Entry Handling ----------------------
            if robot.position(3) <= 10 && ...
               ~arena.isInForestBounds(robot.position(1:2))
                
                fprintf("[NAV] On GROUND outside forest - entering at row 1, col %d\n", target(2));
                
                targetBlock = [];
                targetRow   = min(target(1), 1);
                
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
                    
                    % Boost confidence when entering forest
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
            
            % ------------------ In-Forest Path Execution -------------------
            pathPlanner = PathPlanner(obj.config, obj.arena);
            
            path = pathPlanner.astarWithTerrain( ...
                obj.position, target, ...
                obj.belief, obj.beliefConfidence, obj.mode);
            
            if isempty(path) || size(path, 1) < 2
                fprintf("[NAV] No path to (%d,%d)\n", target);
                return;
            end
            
            nextCell = path(2, :);
            
            fprintf("[NAV DEBUG] Current: (%d,%d), Target: (%d,%d), Next: (%d,%d), PathLen: %d\n", ...
                obj.position(1), obj.position(2), ...
                target(1), target(2), ...
                nextCell(1), nextCell(2), size(path,1));
            
            % Ensure adjacency (safety)
            if abs(nextCell(1) - obj.position(1)) + ...
               abs(nextCell(2) - obj.position(2)) ~= 1
                
                fprintf("[NAV] Non-adjacent step, skipping\n");
                return;
            end
            
            % Resolve block coordinates for next cell
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
            
            % Execute motion
            robot.moveTo([bNext.x, bNext.y], obj.arena);
            obj.position = robot.gridPos;
            obj.pathHistory(end+1, :) = obj.position;
            
            % Boost confidence when stepping onto R2
            if obj.belief(obj.position(1), obj.position(2)) == "R2"
                obj.beliefConfidence(obj.position(1), obj.position(2)) = 0.9;
                fprintf("[NAV] Moved to R2 cell (%d,%d) - boosted confidence to 0.9\n", obj.position);
            else
                fprintf("[NAV] Moved to cell (%d,%d), Robot grid (%d,%d)\n", ...
                    nextCell, robot.gridPos);
            end
        end
        
        
        % ===================================================================
        % Strategy Delegates
        % -------------------------------------------------------------------
        % Thin wrappers around static NavigationPlanner methods.
        % ===================================================================
        function target = chooseBestR2(obj, remainingTime)
            target = NavigationPlanner.chooseBestR2( ...
                obj.position, obj.belief, obj.beliefConfidence, ...
                obj.carry, obj.arena.getHeightMap(), ...
                obj.config.exitCells, remainingTime, obj.config);
        end
        
        function target = findBestExploreCell(obj)
            target = NavigationPlanner.findBestExploreCell( ...
                obj.position, obj.belief, obj.beliefConfidence, ...
                obj.arena.getHeightMap(), obj.visitedCells, obj.config);
        end
        
        function [target, blockedR1] = planExitRoute(obj)
            [target, blockedR1] = NavigationPlanner.planExitRoute( ...
                obj.position, obj.belief, obj.beliefConfidence, ...
                obj.config.exitCells, obj.arena, obj.config);
        end
        
        
        % ===================================================================
        % Reset Navigator State
        % -------------------------------------------------------------------
        % Reinitializes mission state for a fresh run.
        % ===================================================================
        function reset(obj, arena)
            
            obj.mode  = "COLLECT";
            obj.carry = 0;
            
            initialBlock = arena.getInitialBlock();
            
            if initialBlock.row == 0
                obj.position = [1, initialBlock.col];
            else
                obj.position = [initialBlock.row, initialBlock.col];
            end
            
            obj.belief(:)           = "UNSEEN";
            obj.beliefConfidence(:) = 0;
            obj.cellConfirmed(:)    = false;
            obj.visitedCells(:)     = false;
            
            if obj.isInForestGrid(obj.position)
                obj.visitedCells(obj.position(1), obj.position(2)) = true;
                obj.cellConfirmed(obj.position(1), obj.position(2)) = true;
            end
            
            obj.pathHistory      = obj.position;
            obj.pendingTarget    = [];
            obj.confirmWaitStart = [];
            
            obj.gameStartTime = tic;
            obj.emergencyExitTriggered = false;
            
            obj.r1WaitStart    = containers.Map;
            obj.r1WaitAttempts = containers.Map;
            
            obj.lastGridPos   = obj.position;
            obj.stuckCounter  = 0;
            obj.replanCounter = 0;
            
            obj.collectedR2Positions = [];
            obj.pathFailureCount     = 0;
            obj.missionComplete      = false;
            
            fprintf('[NAV] Navigator reset for new mission\n');
        end
        
        
        % ===================================================================
        % Belief Synchronization from Vision Detections
        % -------------------------------------------------------------------
        % Pulls confirmed detections into navigator belief grid.
        % ===================================================================
        function updateBeliefFromDetections(obj, vision, arena)
            
            if isempty(vision.detectedKfs)
                return;
            end
            
            detectedIds = cell2mat(vision.detectedKfs.keys);
            
            for i = 1:length(detectedIds)
                
                kfsId     = detectedIds(i);
                detection = vision.detectedKfs(kfsId);
                
                block = arena.blocks(kfsId);
                row   = block.row;
                col   = block.col;
                
                % Do not overwrite confirmed empty cells
                if obj.belief(row, col) == "EMPTY"
                    continue;
                end
                
                % Update belief and confidence conservatively
                obj.belief(row, col) = detection.type;
                
                obj.beliefConfidence(row, col) = ...
                    max(obj.beliefConfidence(row, col), ...
                        detection.confidence);
            end
        end
    end
end
