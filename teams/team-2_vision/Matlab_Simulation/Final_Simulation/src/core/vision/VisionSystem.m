classdef VisionSystem < handle
    % VISIONSYSTEM Simulates HSV+ML vision pipeline with occlusion
    
    properties
        config
        arena
        
        % Detection state
        detectedKfs         % Map: kfsId -> struct(type, confidence)
        detectionTimers     % Map: kfsId -> tic value
        detectionLog        % Array of detection events
        
        % Stats
        stats
        missionStartTime
    end
    
    methods
        function obj = VisionSystem(config, arena)
            obj.config = config;
            obj.arena = arena;
            obj.detectedKfs = containers.Map('KeyType', 'double', 'ValueType', 'any');
            obj.detectionTimers = containers.Map('KeyType', 'double', 'ValueType', 'uint64');
            obj.detectionLog = struct('time',{},'kfs_id',{},'type',{},'confidence',{},'pos',{});
            obj.missionStartTime = tic;
            obj.stats = struct('r2_found', 0, 'r1_found', 0, 'fake_found', 0);
        end
        
        function detections = processFrame(obj, robotState, arena)
            % Process vision detection for current frame
            cfg = obj.config;
            cameraPos = robotState.position + [0 0 cfg.CAMERA_HEIGHT_OFFSET];
            cameraYaw = robotState.yaw;
            
            yawRad = deg2rad(cameraYaw);
            cameraForward = [cos(yawRad - pi/2), sin(yawRad - pi/2), 0];
            
            detections = [];
            
            for i = 1:numel(arena.kfsIds)
                kfsBlock = arena.blocks(arena.kfsIds(i));
                kfsType = arena.kfsTypes(i);
                kfsId = arena.kfsIds(i);
                
                targetPt = [kfsBlock.x, kfsBlock.y, kfsBlock.h + cfg.KFS_SIZE/2];
                distToKfs = norm(cameraPos(1:2) - targetPt(1:2));
                
                if distToKfs > cfg.DETECTION_RANGE
                    obj.clearTimer(kfsId);
                    continue;
                end
                
                % Check FOV
                toKfs = targetPt(1:2)' - cameraPos(1:2)';
                toKfsNorm = toKfs / norm(toKfs);
                angle = acosd(dot(cameraForward(1:2), toKfsNorm));
                
                if angle > cfg.DETECTION_FOV / 2
                    obj.clearTimer(kfsId);
                    continue;
                end
                
                % Check occlusion
                isVisible = ~OcclusionChecker.isOccluded(cameraPos, targetPt, ...
                    arena.blocks, cfg.BLOCK_SIZE, kfsBlock.id);
                
                if ~isVisible
                    obj.clearTimer(kfsId);
                    continue;
                end
                
                % HSV color detection
                kfsColor = arena.kfsColors{i};
                isBlue = all(abs(kfsColor - cfg.COLOR_KFS_BLUE) < 1e-3);
                
                if ~isBlue
                    obj.clearTimer(kfsId);
                    continue;
                end
                
                % ML confidence simulation
                mlConfidence = 0.6 + 0.3 * (1 - distToKfs / cfg.DETECTION_RANGE);
                mlConfidence = max(0.5, min(0.95, mlConfidence));
                
                % Time-based confirmation
                if ~obj.detectionTimers.isKey(kfsId)
                    obj.detectionTimers(kfsId) = tic;
                end
                
                if toc(obj.detectionTimers(kfsId)) >= cfg.MIN_DETECTION_TIME
                    if ~obj.detectedKfs.isKey(kfsId)
                        obj.registerDetection(kfsId, kfsType, mlConfidence, robotState.position);
                    end
                    
                    % Add to current frame detections
                    detection.id = kfsId;
                    detection.type = kfsType;
                    detection.confidence = mlConfidence;
                    detection.position = targetPt;
                    detections = [detections; detection]; %#ok<AGROW>
                end
            end
        end
        
        function registerDetection(obj, kfsId, kfsType, confidence, robotPos)
            % Register a new KFS detection
            obj.detectedKfs(kfsId) = struct('type', kfsType, 'confidence', confidence);
            
            logEntry = struct( ...
                'time', toc(obj.missionStartTime), ...
                'kfs_id', kfsId, ...
                'type', char(kfsType), ...
                'confidence', confidence, ...
                'pos', robotPos);
            obj.detectionLog(end+1) = logEntry;
            
            % Update stats
            if kfsType == "R2"
                obj.stats.r2_found = obj.stats.r2_found + 1;
            elseif kfsType == "R1"
                obj.stats.r1_found = obj.stats.r1_found + 1;
            else
                obj.stats.fake_found = obj.stats.fake_found + 1;
            end
            
            fprintf('[VISION] Detected %s at Block %d (Conf: %.2f)\n', ...
                    kfsType, kfsId, confidence);
        end
        
        function clearTimer(obj, kfsId)
            if obj.detectionTimers.isKey(kfsId)
                obj.detectionTimers.remove(kfsId);
            end
        end
        
        function reset(obj)
            obj.detectedKfs = containers.Map('KeyType', 'double', 'ValueType', 'any');
            obj.detectionTimers = containers.Map('KeyType', 'double', 'ValueType', 'uint64');
            obj.detectionLog = struct('time',{},'kfs_id',{},'type',{},'confidence',{},'pos',{});
            obj.missionStartTime = tic;
            obj.stats.r2_found = 0;
            obj.stats.r1_found = 0;
            obj.stats.fake_found = 0;
        end
        
        function s = getStats(obj)
            s = obj.stats;
            s.mission_time = toc(obj.missionStartTime);
        end
        
        function [belief, confidence] = updateGridBelief(obj, cameraPos, cameraYaw, ...
                cameraPitch, oldBelief, oldConfidence)
            % Update grid belief based on current camera view
            cfg = obj.config;
            yawRad = deg2rad(cameraYaw);
            camForward = [cos(yawRad - pi/2), sin(yawRad - pi/2), 0];
            
            belief = oldBelief;
            confidence = oldConfidence;
            
            for k = 1:numel(obj.arena.blocks)
                b = obj.arena.blocks(k);
                row = b.row;
                col = b.col;
                
                center3D = [b.x, b.y, b.h + cfg.KFS_SIZE/2];
                dXY = center3D(1:2) - cameraPos(1:2);
                dist = norm(dXY);
                
                if dist > cfg.DETECTION_RANGE
                    continue;
                end
                
                dir2d = dXY / (norm(dXY) + 1e-9);
                angle = acosd(dot(camForward(1:2), dir2d));
                
                if angle > cfg.DETECTION_FOV / 2
                    continue;
                end
                
                if OcclusionChecker.isOccluded(cameraPos, center3D, ...
                        obj.arena.blocks, cfg.BLOCK_SIZE, b.id)
                    continue;
                end
                
                % Update belief based on ground truth
                truthType = obj.arena.trueForest(row, col);
                if truthType == "EMPTY"
                    belief(row, col) = "EMPTY";
                    confidence(row, col) = max(confidence(row, col), 0.9);
                else
                    belief(row, col) = truthType;
                    baseConf = 0.6 + 0.3 * (1 - dist / cfg.DETECTION_RANGE);
                    baseConf = max(0.6, min(0.95, baseConf));
                    confidence(row, col) = max(confidence(row, col), baseConf);
                end
            end
        end
    end
end
