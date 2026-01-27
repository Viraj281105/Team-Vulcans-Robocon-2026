classdef InputHandler < handle
    % INPUTHANDLER Handles keyboard input for the simulation
    
    properties
        ui      % Reference to SimulationUI
        robot   % Will be set during runtime
        arena   % Will be set during runtime
        navigator % Will be set during runtime
        vision  % Will be set during runtime
    end
    
    methods
        function obj = InputHandler(ui)
            obj.ui = ui;
        end
        
        function setReferences(obj, robot, arena, navigator, vision)
            obj.robot = robot;
            obj.arena = arena;
            obj.navigator = navigator;
            obj.vision = vision;
        end
        
        function handleKey(obj, ~, event)
            % Main keyboard handler
            cfg = obj.ui.config;
            
            switch event.Key
                case 'escape'
                    obj.ui.running = false;
                    
                case 'h'
                    obj.ui.showHelp = ~obj.ui.showHelp;
                    
                case 'v'
                    obj.ui.autoTiltEnabled = ~obj.ui.autoTiltEnabled;
                    fprintf('[AUTO-TILT] %s\n', string(obj.ui.autoTiltEnabled));
                    
                case 'n'
                    obj.ui.autonomousMode = ~obj.ui.autonomousMode;
                    fprintf('[NAV] Autonomous mode %d\n', obj.ui.autonomousMode);
                    
                case 'p'
                    if ~isempty(obj.robot)
                        obj.robot.setRecording(~obj.robot.recordingPath);
                        fprintf('[PATH] Recording %s\n', string(obj.robot.recordingPath));
                    end
                    
                case 't'
                    if ~isempty(obj.arena)
                        obj.arena.randomizeKFS();
                        if ~isempty(obj.vision)
                            obj.vision.reset();
                        end
                    end
                    
                case 'l'
                    obj.exportTrajectory();
                    
                case 'r'
                    if ~isempty(obj.robot) && ~isempty(obj.arena)
                        obj.robot.reset(obj.arena);
                        obj.ui.cameraYawOffset = 0;
                        obj.ui.cameraPitch = 15;
                        if ~isempty(obj.vision)
                            obj.vision.reset();
                        end
                        if ~isempty(obj.navigator)
                            obj.navigator.reset(obj.arena);
                        end
                    end
                    
                % Manual movement (only when not autonomous)
                case 'w'
                    if ~obj.ui.autonomousMode && ~isempty(obj.robot)
                        obj.robot.moveForward(cfg.MOVE_SPEED, obj.arena);
                    end
                    
                case 's'
                    if ~obj.ui.autonomousMode && ~isempty(obj.robot)
                        obj.robot.moveBackward(cfg.MOVE_SPEED, obj.arena);
                    end
                    
                case 'a'
                    if ~obj.ui.autonomousMode && ~isempty(obj.robot)
                        obj.robot.strafeLeft(cfg.MOVE_SPEED, obj.arena);
                    end
                    
                case 'd'
                    if ~obj.ui.autonomousMode && ~isempty(obj.robot)
                        obj.robot.strafeRight(cfg.MOVE_SPEED, obj.arena);
                    end
                    
                case 'space'
                    if ~obj.ui.autonomousMode && ~isempty(obj.robot)
                        obj.robot.climbToBlock(obj.arena);
                    end
                    
                case 'c'
                    if ~obj.ui.autonomousMode && ~isempty(obj.robot)
                        obj.robot.descend();
                    end
                    
                case 'q'
                    if ~obj.ui.autonomousMode && ~isempty(obj.robot)
                        obj.robot.rotate(cfg.ROTATION_SPEED);
                    end
                    
                case 'e'
                    if ~obj.ui.autonomousMode && ~isempty(obj.robot)
                        obj.robot.rotate(-cfg.ROTATION_SPEED);
                    end
                    
                % Camera controls
                case 'i'
                    obj.ui.cameraYawOffset = obj.ui.cameraYawOffset + cfg.YAW_STEP;
                    fprintf('[CAMERA] Pan: %.0f°\n', obj.ui.cameraYawOffset);
                    
                case 'k'
                    obj.ui.cameraYawOffset = obj.ui.cameraYawOffset - cfg.YAW_STEP;
                    fprintf('[CAMERA] Pan: %.0f°\n', obj.ui.cameraYawOffset);
                    
                case 'z'
                    if ~obj.ui.autoTiltEnabled
                        obj.ui.cameraPitch = max(-45, obj.ui.cameraPitch - cfg.PITCH_STEP);
                        fprintf('[CAMERA] Pitch: %.0f°\n', obj.ui.cameraPitch);
                    end
                    
                case 'x'
                    if ~obj.ui.autoTiltEnabled
                        obj.ui.cameraPitch = min(45, obj.ui.cameraPitch + cfg.PITCH_STEP);
                        fprintf('[CAMERA] Pitch: %.0f°\n', obj.ui.cameraPitch);
                    end
            end
        end
        
        function exportTrajectory(obj)
            % Export trajectory and mission data
            if isempty(obj.robot) || isempty(obj.vision)
                fprintf('[EXPORT] No data to export\n');
                return;
            end
            
            timestamp = datestr(now, 'yyyymmdd_HHMMSS');
            
            % Export path
            if ~isempty(obj.robot.pathHistory)
                csvFile = sprintf('trajectory_%s.csv', timestamp);
                csvwrite(csvFile, obj.robot.pathHistory);
                fprintf('[EXPORT] Path: %s\n', csvFile);
            end
            
            % Export mission data
            matFile = sprintf('mission_data_%s.mat', timestamp);
            pathHistory = obj.robot.pathHistory;
            detectionLog = obj.vision.detectionLog;
            stats = obj.vision.getStats();
            stats.total_distance = obj.robot.getTotalDistance();
            
            if ~isempty(obj.arena)
                kfsIds = obj.arena.kfsIds;
                kfsTypes = obj.arena.kfsTypes;
                kfsColors = obj.arena.kfsColors;
                save(matFile, 'pathHistory', 'detectionLog', 'stats', ...
                    'kfsIds', 'kfsTypes', 'kfsColors');
            else
                save(matFile, 'pathHistory', 'detectionLog', 'stats');
            end
            
            fprintf('[EXPORT] Data: %s\n', matFile);
            
            % Print summary
            fprintf('\n=== MISSION SUMMARY ===\n');
            fprintf('Duration: %.1fs\n', stats.mission_time);
            fprintf('Distance: %.2fm\n', stats.total_distance/1000);
            fprintf('R2: %d/%d | R1: %d/%d | Fake: %d/%d\n', ...
                stats.r2_found, obj.ui.config.TOTAL_R2_REAL, ...
                stats.r1_found, obj.ui.config.TOTAL_R1, ...
                stats.fake_found, obj.ui.config.TOTAL_FAKE);
            fprintf('Detections: %d\n', numel(detectionLog));
            fprintf('======================\n\n');
        end
    end
end
