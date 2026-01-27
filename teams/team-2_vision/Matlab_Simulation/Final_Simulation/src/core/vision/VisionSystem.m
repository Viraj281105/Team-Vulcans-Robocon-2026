classdef VisionSystem < handle
    % ===================================================================
    % VISIONSYSTEM
    % -------------------------------------------------------------------
    % Simulates a hybrid HSV + ML vision pipeline with:
    %   - Geometric visibility checks
    %   - Field-of-view filtering
    %   - Occlusion reasoning
    %   - Time-based detection confirmation
    %   - Confidence modeling
    %   - Detection logging and statistics
    %   - Grid-based belief propagation
    %
    % Architectural role:
    %   - Acts as the perception subsystem of the simulator.
    %   - Converts raw geometry + robot pose into semantic detections.
    %   - Maintains memory of detected objects and confidence levels.
    %   - Provides probabilistic belief updates for navigation.
    %
    % This class intentionally does NOT:
    %   - Control robot motion
    %   - Perform rendering
    %   - Perform planning or navigation
    %
    % Think of this as a realistic abstraction of a robotics vision stack.
    % ===================================================================
    
    properties
        % ------------------ Shared References ---------------------------
        config                  % Global simulation configuration
        arena                   % Arena containing blocks and KFS layout
        
        % ------------------ Detection State -----------------------------
        detectedKfs             % Map: kfsId -> struct(type, confidence)
        detectionTimers         % Map: kfsId -> tic value (for debounce)
        detectionLog            % Array of detection events (struct array)
        
        % ------------------ Statistics & Timing -------------------------
        stats                   % Counters for detected object types
        missionStartTime        % tic reference for elapsed mission time
    end
    
    methods
        
        % ===================================================================
        % Constructor
        % -------------------------------------------------------------------
        % Initializes internal state containers and statistics.
        % ===================================================================
        function obj = VisionSystem(config, arena)
            
            % Store shared references
            obj.config = config;
            obj.arena  = arena;
            
            % Map storing confirmed detections
            %   Key   → kfsId (double)
            %   Value → struct('type', ..., 'confidence', ...)
            obj.detectedKfs = containers.Map( ...
                'KeyType', 'double', 'ValueType', 'any');
            
            % Map storing timers used for time-based confirmation
            %   Each key holds the tic timestamp when detection started
            obj.detectionTimers = containers.Map( ...
                'KeyType', 'double', 'ValueType', 'uint64');
            
            % Detection log stores every confirmed detection event
            obj.detectionLog = struct( ...
                'time', {}, 'kfs_id', {}, ...
                'type', {}, 'confidence', {}, 'pos', {});
            
            % Start mission timer
            obj.missionStartTime = tic;
            
            % Initialize detection counters
            obj.stats = struct( ...
                'r2_found', 0, ...
                'r1_found', 0, ...
                'fake_found', 0);
        end
        
        
        % ===================================================================
        % Per-Frame Vision Processing
        % -------------------------------------------------------------------
        % Simulates vision detection for a single frame.
        % ===================================================================
        function detections = processFrame(obj, robotState, arena)
            
            cfg = obj.config;
            
            % Compute camera position in world coordinates
            % Camera is vertically offset above robot base.
            cameraPos = robotState.position + ...
                        [0 0 cfg.CAMERA_HEIGHT_OFFSET];
            
            % Camera yaw follows robot yaw
            cameraYaw = robotState.yaw;
            
            % Convert yaw to radians
            yawRad = deg2rad(cameraYaw);
            
            % Compute forward direction vector of camera (XY plane)
            cameraForward = ...
                [cos(yawRad - pi/2), sin(yawRad - pi/2), 0];
            
            % Initialize empty detection list for this frame
            detections = [];
            
            % ---------------------------------------------------------------
            % Iterate over every KFS object in the arena
            % ---------------------------------------------------------------
            for i = 1:numel(arena.kfsIds)
                
                % Retrieve KFS block and metadata
                kfsBlock = arena.blocks(arena.kfsIds(i));
                kfsType  = arena.kfsTypes(i);
                kfsId    = arena.kfsIds(i);
                
                % Compute center point of KFS cube
                targetPt = ...
                    [kfsBlock.x, ...
                     kfsBlock.y, ...
                     kfsBlock.h + cfg.KFS_SIZE/2];
                
                % Euclidean distance in XY plane
                distToKfs = ...
                    norm(cameraPos(1:2) - targetPt(1:2));
                
                % -----------------------------------------------------------
                % Range gating
                % -----------------------------------------------------------
                if distToKfs > cfg.DETECTION_RANGE
                    obj.clearTimer(kfsId);
                    continue;
                end
                
                % -----------------------------------------------------------
                % Field-of-view gating
                % -----------------------------------------------------------
                toKfs = targetPt(1:2)' - cameraPos(1:2)';
                toKfsNorm = toKfs / norm(toKfs);
                angle = acosd(dot(cameraForward(1:2), toKfsNorm));
                
                if angle > cfg.DETECTION_FOV / 2
                    obj.clearTimer(kfsId);
                    continue;
                end
                
                % -----------------------------------------------------------
                % Occlusion test using ray casting
                % -----------------------------------------------------------
                isVisible = ...
                    ~OcclusionChecker.isOccluded( ...
                        cameraPos, targetPt, ...
                        arena.blocks, cfg.BLOCK_SIZE, ...
                        kfsBlock.id);
                
                if ~isVisible
                    obj.clearTimer(kfsId);
                    continue;
                end
                
                % -----------------------------------------------------------
                % HSV color filtering (simulated)
                % Only blue KFS are considered valid detections.
                % -----------------------------------------------------------
                kfsColor = arena.kfsColors{i};
                isBlue = ...
                    all(abs(kfsColor - cfg.COLOR_KFS_BLUE) < 1e-3);
                
                if ~isBlue
                    obj.clearTimer(kfsId);
                    continue;
                end
                
                % -----------------------------------------------------------
                % Simulated ML confidence model
                % Confidence decreases with distance.
                % -----------------------------------------------------------
                mlConfidence = ...
                    0.6 + 0.3 * (1 - distToKfs / cfg.DETECTION_RANGE);
                
                % Clamp confidence into realistic bounds
                mlConfidence = ...
                    max(0.5, min(0.95, mlConfidence));
                
                % -----------------------------------------------------------
                % Time-based confirmation filter
                % Ensures object must persist in view for a minimum duration.
                % -----------------------------------------------------------
                if ~obj.detectionTimers.isKey(kfsId)
                    obj.detectionTimers(kfsId) = tic;
                end
                
                if toc(obj.detectionTimers(kfsId)) >= ...
                        cfg.MIN_DETECTION_TIME
                    
                    % Register detection only once
                    if ~obj.detectedKfs.isKey(kfsId)
                        obj.registerDetection( ...
                            kfsId, kfsType, ...
                            mlConfidence, robotState.position);
                    end
                    
                    % Add detection to current frame output
                    detection.id         = kfsId;
                    detection.type       = kfsType;
                    detection.confidence = mlConfidence;
                    detection.position   = targetPt;
                    
                    detections = ...
                        [detections; detection]; %#ok<AGROW>
                end
            end
        end
        
        
        % ===================================================================
        % Register Confirmed Detection
        % -------------------------------------------------------------------
        % Stores detection permanently and updates statistics.
        % ===================================================================
        function registerDetection(obj, kfsId, kfsType, ...
                                   confidence, robotPos)
            
            % Store detection in map
            obj.detectedKfs(kfsId) = ...
                struct('type', kfsType, 'confidence', confidence);
            
            % Create log entry
            logEntry = struct( ...
                'time', toc(obj.missionStartTime), ...
                'kfs_id', kfsId, ...
                'type', char(kfsType), ...
                'confidence', confidence, ...
                'pos', robotPos);
            
            % Append to detection history
            obj.detectionLog(end+1) = logEntry;
            
            % Update counters based on type
            if kfsType == "R2"
                obj.stats.r2_found = obj.stats.r2_found + 1;
            elseif kfsType == "R1"
                obj.stats.r1_found = obj.stats.r1_found + 1;
            else
                obj.stats.fake_found = obj.stats.fake_found + 1;
            end
            
            % Console feedback
            fprintf('[VISION] Detected %s at Block %d (Conf: %.2f)\n', ...
                    kfsType, kfsId, confidence);
        end
        
        
        % ===================================================================
        % Clear Detection Timer
        % -------------------------------------------------------------------
        % Resets confirmation timer when object is no longer visible.
        % ===================================================================
        function clearTimer(obj, kfsId)
            if obj.detectionTimers.isKey(kfsId)
                obj.detectionTimers.remove(kfsId);
            end
        end
        
        
        % ===================================================================
        % Reset Vision State
        % -------------------------------------------------------------------
        % Clears all memory and restarts statistics and timers.
        % ===================================================================
        function reset(obj)
            
            % Reset detection memory
            obj.detectedKfs = containers.Map( ...
                'KeyType', 'double', 'ValueType', 'any');
            
            % Reset timers
            obj.detectionTimers = containers.Map( ...
                'KeyType', 'double', 'ValueType', 'uint64');
            
            % Reset detection log
            obj.detectionLog = struct( ...
                'time', {}, 'kfs_id', {}, ...
                'type', {}, 'confidence', {}, 'pos', {});
            
            % Restart mission timer
            obj.missionStartTime = tic;
            
            % Reset statistics
            obj.stats.r2_found   = 0;
            obj.stats.r1_found   = 0;
            obj.stats.fake_found = 0;
        end
        
        
        % ===================================================================
        % Retrieve Statistics Snapshot
        % -------------------------------------------------------------------
        % Returns current statistics with mission time injected.
        % ===================================================================
        function s = getStats(obj)
            s = obj.stats;
            s.mission_time = toc(obj.missionStartTime);
        end
        
        
        % ===================================================================
        % Grid Belief Update
        % -------------------------------------------------------------------
        % Updates probabilistic belief grid using camera visibility.
        % ===================================================================
        function [belief, confidence] = updateGridBelief( ...
                obj, cameraPos, cameraYaw, cameraPitch, ...
                oldBelief, oldConfidence)
            
            cfg = obj.config;
            
            % Compute camera forward vector
            yawRad = deg2rad(cameraYaw);
            camForward = ...
                [cos(yawRad - pi/2), sin(yawRad - pi/2), 0];
            
            % Initialize outputs with previous state
            belief     = oldBelief;
            confidence = oldConfidence;
            
            % ---------------------------------------------------------------
            % Iterate through every block in arena grid
            % ---------------------------------------------------------------
            for k = 1:numel(obj.arena.blocks)
                
                b   = obj.arena.blocks(k);
                row = b.row;
                col = b.col;
                
                % Center of block volume
                center3D = ...
                    [b.x, b.y, b.h + cfg.KFS_SIZE/2];
                
                % Distance from camera
                dXY  = center3D(1:2) - cameraPos(1:2);
                dist = norm(dXY);
                
                % Range gating
                if dist > cfg.DETECTION_RANGE
                    continue;
                end
                
                % Angular gating
                dir2d = dXY / (norm(dXY) + 1e-9);
                angle = acosd(dot(camForward(1:2), dir2d));
                
                if angle > cfg.DETECTION_FOV / 2
                    continue;
                end
                
                % Occlusion test
                if OcclusionChecker.isOccluded( ...
                        cameraPos, center3D, ...
                        obj.arena.blocks, ...
                        cfg.BLOCK_SIZE, b.id)
                    continue;
                end
                
                % -----------------------------------------------------------
                % Belief update using ground truth (simulation shortcut)
                % -----------------------------------------------------------
                truthType = obj.arena.trueForest(row, col);
                
                if truthType == "EMPTY"
                    belief(row, col)     = "EMPTY";
                    confidence(row, col) = ...
                        max(confidence(row, col), 0.9);
                else
                    belief(row, col) = truthType;
                    
                    baseConf = ...
                        0.6 + 0.3 * (1 - dist / cfg.DETECTION_RANGE);
                    
                    baseConf = ...
                        max(0.6, min(0.95, baseConf));
                    
                    confidence(row, col) = ...
                        max(confidence(row, col), baseConf);
                end
            end
        end
    end
end
