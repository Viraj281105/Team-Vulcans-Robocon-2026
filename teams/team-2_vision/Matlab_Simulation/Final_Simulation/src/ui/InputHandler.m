classdef InputHandler < handle
    % ===================================================================
    % INPUTHANDLER
    % -------------------------------------------------------------------
    % Centralized keyboard input processing engine for the simulation.
    %
    % Architectural role:
    %   - Acts as the controller layer between UI events and system actions.
    %   - Receives raw keyboard events from SimulationUI.
    %   - Translates keystrokes into deterministic operations on:
    %         • Robot motion and pose
    %         • Arena randomization
    %         • Vision resets
    %         • Navigation mode toggles
    %         • Camera manipulation
    %         • Data export pipelines
    %
    % Design philosophy:
    %   - Keeps all input logic in one place (single-responsibility).
    %   - Prevents UI rendering code from becoming behavior-heavy.
    %   - Explicitly guards against null references for safety.
    %
    % This class does NOT:
    %   - Render graphics
    %   - Perform planning
    %   - Execute perception logic
    % ===================================================================
    
    properties
        % ------------------ Core References -----------------------------
        ui          % Reference to SimulationUI (always available)
        robot       % Robot object (injected after construction)
        arena       % Arena object (injected after construction)
        navigator   % Navigator object (injected after construction)
        vision      % VisionSystem object (injected after construction)
    end
    
    methods
        
        % ===================================================================
        % Constructor
        % -------------------------------------------------------------------
        % Stores UI reference for accessing configuration and UI state.
        % ===================================================================
        function obj = InputHandler(ui)
            obj.ui = ui;
        end
        
        
        % ===================================================================
        % Reference Injection
        % -------------------------------------------------------------------
        % Called after all subsystems are created to wire dependencies.
        % ===================================================================
        function setReferences(obj, robot, arena, navigator, vision)
            obj.robot     = robot;
            obj.arena     = arena;
            obj.navigator = navigator;
            obj.vision    = vision;
        end
        
        
        % ===================================================================
        % Keyboard Event Dispatcher
        % -------------------------------------------------------------------
        % Maps key presses to system actions.
        % ===================================================================
        function handleKey(obj, ~, event)
            
            % Shorthand access to configuration parameters
            cfg = obj.ui.config;
            
            % ---------------------------------------------------------------
            % Key mapping table implemented as switch-case
            % ---------------------------------------------------------------
            switch event.Key
                
                % ------------------ Global Exit ---------------------------
                case 'escape'
                    % Gracefully terminate the simulation loop
                    obj.ui.running = false;
                    
                % ------------------ Help Toggle ---------------------------
                case 'h'
                    % Toggle visibility of help overlay
                    obj.ui.showHelp = ~obj.ui.showHelp;
                    
                % ------------------ Camera Auto-Tilt Toggle ---------------
                case 'v'
                    obj.ui.autoTiltEnabled = ~obj.ui.autoTiltEnabled;
                    fprintf('[AUTO-TILT] %s\n', string(obj.ui.autoTiltEnabled));
                    
                % ------------------ Autonomous Navigation Toggle ----------
                case 'n'
                    obj.ui.autonomousMode = ~obj.ui.autonomousMode;
                    fprintf('[NAV] Autonomous mode %d\n', obj.ui.autonomousMode);
                    
                % ------------------ Path Recording Toggle -----------------
                case 'p'
                    if ~isempty(obj.robot)
                        obj.robot.setRecording(~obj.robot.recordingPath);
                        fprintf('[PATH] Recording %s\n', ...
                                string(obj.robot.recordingPath));
                    end
                    
                % ------------------ Randomize KFS Layout ------------------
                case 't'
                    if ~isempty(obj.arena)
                        obj.arena.randomizeKFS();
                        
                        % Reset vision state to match new environment
                        if ~isempty(obj.vision)
                            obj.vision.reset();
                        end
                    end
                    
                % ------------------ Export Trajectory ---------------------
                case 'l'
                    obj.exportTrajectory();
                    
                % ------------------ Full System Reset ---------------------
                case 'r'
                    if ~isempty(obj.robot) && ~isempty(obj.arena)
                        
                        % Reset robot pose and internal state
                        obj.robot.reset(obj.arena);
                        
                        % Reset camera orientation offsets
                        obj.ui.cameraYawOffset = 0;
                        obj.ui.cameraPitch     = 15;
                        
                        % Reset perception memory
                        if ~isempty(obj.vision)
                            obj.vision.reset();
                        end
                        
                        % Reset navigation planner state
                        if ~isempty(obj.navigator)
                            obj.navigator.reset(obj.arena);
                        end
                    end
                    
                % ===========================================================
                % Manual Robot Movement (disabled during autonomous mode)
                % ===========================================================
                
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
                    
                % ===========================================================
                % Camera Orientation Controls
                % ===========================================================
                
                case 'i'
                    % Pan camera to the left (positive yaw offset)
                    obj.ui.cameraYawOffset = ...
                        obj.ui.cameraYawOffset + cfg.YAW_STEP;
                    fprintf('[CAMERA] Pan: %.0f°\n', obj.ui.cameraYawOffset);
                    
                case 'k'
                    % Pan camera to the right (negative yaw offset)
                    obj.ui.cameraYawOffset = ...
                        obj.ui.cameraYawOffset - cfg.YAW_STEP;
                    fprintf('[CAMERA] Pan: %.0f°\n', obj.ui.cameraYawOffset);
                    
                case 'z'
                    % Manual pitch down (disabled when auto-tilt enabled)
                    if ~obj.ui.autoTiltEnabled
                        obj.ui.cameraPitch = ...
                            max(-45, obj.ui.cameraPitch - cfg.PITCH_STEP);
                        fprintf('[CAMERA] Pitch: %.0f°\n', obj.ui.cameraPitch);
                    end
                    
                case 'x'
                    % Manual pitch up (disabled when auto-tilt enabled)
                    if ~obj.ui.autoTiltEnabled
                        obj.ui.cameraPitch = ...
                            min(45, obj.ui.cameraPitch + cfg.PITCH_STEP);
                        fprintf('[CAMERA] Pitch: %.0f°\n', obj.ui.cameraPitch);
                    end
            end
        end
        
        
        % ===================================================================
        % Export Trajectory and Mission Data
        % -------------------------------------------------------------------
        % Serializes run data for offline analysis and replay.
        % ===================================================================
        function exportTrajectory(obj)
            
            % Guard against missing references
            if isempty(obj.robot) || isempty(obj.vision)
                fprintf('[EXPORT] No data to export\n');
                return;
            end
            
            % Timestamp used for unique file naming
            timestamp = datestr(now, 'yyyymmdd_HHMMSS');
            
            % ------------------ Export Path History ----------------------
            if ~isempty(obj.robot.pathHistory)
                csvFile = sprintf('trajectory_%s.csv', timestamp);
                csvwrite(csvFile, obj.robot.pathHistory);
                fprintf('[EXPORT] Path: %s\n', csvFile);
            end
            
            % ------------------ Export Mission Metadata ------------------
            matFile = sprintf('mission_data_%s.mat', timestamp);
            
            pathHistory  = obj.robot.pathHistory;
            detectionLog = obj.vision.detectionLog;
            stats         = obj.vision.getStats();
            
            % Attach derived statistics
            stats.total_distance = obj.robot.getTotalDistance();
            
            % Include arena metadata if available
            if ~isempty(obj.arena)
                kfsIds    = obj.arena.kfsIds;
                kfsTypes  = obj.arena.kfsTypes;
                kfsColors = obj.arena.kfsColors;
                
                save(matFile, ...
                    'pathHistory', 'detectionLog', 'stats', ...
                    'kfsIds', 'kfsTypes', 'kfsColors');
            else
                save(matFile, ...
                    'pathHistory', 'detectionLog', 'stats');
            end
            
            fprintf('[EXPORT] Data: %s\n', matFile);
            
            % ------------------ Console Summary --------------------------
            fprintf('\n=== MISSION SUMMARY ===\n');
            fprintf('Duration: %.1fs\n', stats.mission_time);
            fprintf('Distance: %.2fm\n', stats.total_distance/1000);
            fprintf('R2: %d/%d | R1: %d/%d | Fake: %d/%d\n', ...
                stats.r2_found,   obj.ui.config.TOTAL_R2_REAL, ...
                stats.r1_found,   obj.ui.config.TOTAL_R1, ...
                stats.fake_found, obj.ui.config.TOTAL_FAKE);
            fprintf('Detections: %d\n', numel(detectionLog));
            fprintf('======================\n\n');
        end
    end
end
