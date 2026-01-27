classdef SimulationUI < handle
    % ===================================================================
    % SIMULATIONUI
    % -------------------------------------------------------------------
    % This class manages the entire graphical user interface (GUI) layer
    % of the ROBOCON simulation.
    %
    % Core responsibilities:
    %   - Create and configure MATLAB figures and axes.
    %   - Render the 3D arena overview and camera feed.
    %   - Display HUD overlays and telemetry.
    %   - Handle keyboard input and dispatch commands.
    %   - Maintain UI runtime state (help, auto-tilt, autonomy, etc.).
    %
    % Architectural role:
    %   - Acts as the visualization and interaction layer.
    %   - Does NOT contain simulation logic or navigation logic.
    %   - Delegates motion commands to InputHandler.
    %   - Consumes state from Robot, Arena, Vision, Navigator.
    %
    % This separation ensures clean MVC-style architecture:
    %   Model   → Robot / Arena / Navigator / Vision
    %   View    → SimulationUI
    %   Control → InputHandler
    % ===================================================================

    properties
        % ------------------ Shared System References -------------------
        config              % Global simulation configuration object
        arena               % Arena/environment object
        
        % ------------------ Figure and Axes Handles --------------------
        fig                 % Main MATLAB figure window
        axMain              % Left panel: 3D arena overview
        axCam               % Right panel: simulated camera feed
        
        % ------------------ UI Runtime State Flags ---------------------
        running             % True while simulation should keep running
        showHelp            % Toggle for help overlay visibility
        autoTiltEnabled     % Enables automatic camera pitch adjustment
        autonomousMode      % Enables autonomous navigation
        
        % ------------------ Camera Control State -----------------------
        cameraPitch         % Camera vertical tilt angle (degrees)
        cameraYawOffset     % Additional yaw offset applied to robot yaw
        
        % ------------------ Input System -------------------------------
        inputHandler        % Object responsible for interpreting key input
    end
    
    methods
        
        % ===================================================================
        % Constructor
        % -------------------------------------------------------------------
        % Initializes UI state, creates figures, and binds input handlers.
        % ===================================================================
        function obj = SimulationUI(config, arena)
            
            % Store shared references for later rendering and querying
            obj.config = config;
            obj.arena = arena;
            
            % ------------------ Default UI State --------------------------
            obj.running = true;             % Simulation is active initially
            obj.showHelp = false;           % Help overlay hidden by default
            obj.autoTiltEnabled = false;    % Auto camera tilt disabled
            obj.autonomousMode = false;     % Manual control initially
            
            % ------------------ Initial Camera State ---------------------
            obj.cameraPitch = 15;           % Default downward tilt (degrees)
            obj.cameraYawOffset = 0;        % No yaw offset initially
            
            % Create figure window and axes layout
            obj.setupFigure();
            
            % Create input handler and bind it to this UI instance
            obj.inputHandler = InputHandler(obj);
            
            % Inform user that keyboard capture is active
            % This is important because MATLAB figures lose focus easily.
            fprintf('[UI] Keyboard controls enabled - figure must have focus\n');
        end
        
        
        % ===================================================================
        % Figure and Axes Setup
        % -------------------------------------------------------------------
        % Creates the main figure window and subplot layout.
        % ===================================================================
        function setupFigure(obj)
            
            % Create main figure window with custom styling and callbacks
            obj.fig = figure( ...
                'Name', 'ROBOCON 2026 - 3D + NAV Simulation', ...
                'Position', [50 50 1800 900], ...              % Large widescreen window
                'Color', [0.15 0.15 0.15], ...                % Dark UI theme
                'NumberTitle', 'off', ...
                'MenuBar', 'none', ...
                'ToolBar', 'figure', ...
                'WindowKeyPressFcn', @(src, event) obj.handleKeyPress(src, event));  
                % WindowKeyPressFcn ensures keyboard input is captured reliably
            
            % Create subplot layout:
            %   Left side  → Arena overview (axMain)
            %   Right side → Camera view   (axCam)
            obj.axMain = subplot(2, 3, [1 4]);
            obj.axCam  = subplot(2, 3, [2 3 5 6]);
            
            % Ensure the figure can be interrupted by keyboard callbacks
            % even while rendering is occurring.
            set(obj.fig, 'BusyAction', 'cancel', 'Interruptible', 'on');
        end
        
        
        % ===================================================================
        % Keyboard Event Handler
        % -------------------------------------------------------------------
        % Central dispatch function for all keyboard input.
        % ===================================================================
        function handleKeyPress(obj, ~, event)
            
            % Keys that should be ignored (system-level modifiers)
            % This prevents accidental blocking of OS shortcuts.
            systemKeys = {'alt', 'control', 'shift', 'tab', ...
                          'windows', 'command', 'capslock'};
            
            % Ignore modifier keys silently
            if ismember(lower(event.Key), systemKeys)
                return;  
            end
            
            % Debug logging for development and diagnostics
            fprintf('[UI DEBUG] Key pressed: %s\n', event.Key);
            
            % ---------------------------------------------------------------
            % UI-only controls handled directly here
            % These keys should NOT be forwarded to InputHandler
            % ---------------------------------------------------------------
            switch event.Key
                
                case 'n'
                    % Toggle autonomous navigation mode
                    obj.autonomousMode = ~obj.autonomousMode;
                    fprintf('[NAV] Autonomous mode %d\n', obj.autonomousMode);
                    return;     % Prevent duplicate processing
                    
                case 'h'
                    % Toggle help overlay visibility
                    obj.showHelp = ~obj.showHelp;
                    fprintf('[UI] Help overlay %d\n', obj.showHelp);
                    return;
                    
                case 'v'
                    % Toggle automatic camera pitch tracking
                    obj.autoTiltEnabled = ~obj.autoTiltEnabled;
                    fprintf('[CAMERA] Auto-tilt %d\n', obj.autoTiltEnabled);
                    return;
                    
                case 'escape'
                    % Graceful shutdown of simulation
                    obj.running = false;
                    fprintf('[UI] Simulation ending...\n');
                    close(obj.fig);
                    return;
            end
            
            % ---------------------------------------------------------------
            % Robot control keys are delegated to InputHandler
            % ---------------------------------------------------------------
            if ~isempty(obj.inputHandler)
                obj.inputHandler.handleKey([], event);
            end
        end


        % ===================================================================
        % Main Render Dispatcher
        % -------------------------------------------------------------------
        % Called once per simulation frame.
        % Coordinates all rendering operations.
        % ===================================================================
        function render(obj, robot, arena, vision, navigator, detections)
            
            % Abort rendering if simulation is no longer active
            if ~obj.running || ~ishandle(obj.fig)
                return;
            end
            
            % Ensure the figure remains focused for keyboard input
            if ~strcmp(get(obj.fig, 'CurrentObject'), '')
                figure(obj.fig);  
            end
            
            % Fetch latest robot state snapshot
            robotState = robot.getState();
            
            % Automatically adjust camera pitch if enabled
            if obj.autoTiltEnabled
                obj.cameraPitch = obj.autoAdjustPitch(robotState, arena, detections);
            end
            
            % Render left-side arena overview
            obj.renderOverview(robot, arena, vision, navigator);
            
            % Render right-side camera view
            obj.renderCameraView(robot, arena, vision, detections, navigator);
            
            % Flush graphics pipeline with throttling for performance
            drawnow limitrate;
        end
        
        
        % ===================================================================
        % Arena Overview Renderer (Left Panel)
        % -------------------------------------------------------------------
        % Displays the full 3D environment, robot pose, paths, and HUD.
        % ===================================================================
        function renderOverview(obj, robot, arena, vision, navigator)
            
            cfg = obj.config;
            
            % Activate the main axes
            set(obj.fig, 'CurrentAxes', obj.axMain);
            cla(obj.axMain);
            hold(obj.axMain, 'on');
            grid(obj.axMain, 'on');
            axis(obj.axMain, 'equal');
            
            % Axis labeling and viewing angle
            xlabel(obj.axMain, 'X (mm)');
            ylabel(obj.axMain, 'Y (mm)');
            zlabel(obj.axMain, 'Z (mm)');
            view(obj.axMain, 45, 30);
            
            % World bounds
            xlim(obj.axMain, [0 cfg.ARENA_X]);
            ylim(obj.axMain, [0 cfg.ARENA_Y]);
            zlim(obj.axMain, [0 cfg.ARENA_Z]);
            
            % Dark background for contrast
            set(obj.axMain, 'Color', [0.05 0.05 0.1]);
            
            % ------------------ Ground Plane ------------------------------
            patch(obj.axMain, ...
                [0 cfg.ARENA_X cfg.ARENA_X 0], ...
                [0 0 cfg.ARENA_Y cfg.ARENA_Y], ...
                [0 0 0 0], ...
                cfg.COLOR_PATHWAY, ...
                'FaceAlpha', 0.3, 'EdgeColor', 'none');
            
            % ------------------ Arena Blocks ------------------------------
            Renderer.drawBlocks(obj.axMain, arena.blocks, ...
                                robot.currentBlockId, cfg);
            
            % ------------------ KFS Objects -------------------------------
            Renderer.drawKFS(obj.axMain, arena, vision.detectedKfs, cfg);
            
            % ------------------ Robot Visualization -----------------------
            Renderer.drawRobot(obj.axMain, robot.getState(), cfg);
            
            % ------------------ Detection Range ---------------------------
            Renderer.drawDetectionRange(obj.axMain, robot.position, cfg);
            
            % ------------------ Path History ------------------------------
            if ~isempty(robot.pathHistory)
                plot3(obj.axMain, ...
                    robot.pathHistory(:,1), ...
                    robot.pathHistory(:,2), ...
                    robot.pathHistory(:,3), ...
                    'g-', 'LineWidth', 2);
            end
            
            % ------------------ Telemetry Title ---------------------------
            title(obj.axMain, sprintf( ...
                'Mission: %.1fs | Dist: %.1fm | Yaw: %.0f° | Carry: %d/%d | Auto: %d', ...
                toc(navigator.gameStartTime), ...
                robot.getTotalDistance()/1000, ...
                robot.yaw, ...
                navigator.carry, cfg.CAPACITY, ...
                obj.autonomousMode), ...
                'Color', 'white');
            
            % ------------------ Help Overlay ------------------------------
            if obj.showHelp
                obj.renderHelpText();
            end
        end
        
        
        % ===================================================================
        % Camera View Renderer (Right Panel)
        % -------------------------------------------------------------------
        % Simulates the robot-mounted camera perspective.
        % ===================================================================
        function renderCameraView(obj, robot, arena, vision, detections, navigator)
            
            cfg = obj.config;
            robotState = robot.getState();
            
            % Compute camera world position
            cameraPos = robotState.position + ...
                        [0 0 cfg.CAMERA_HEIGHT_OFFSET];
            
            % Compute effective yaw (robot yaw + UI offset)
            cameraYaw = robotState.yaw + obj.cameraYawOffset;
            
            % Activate camera axes
            set(obj.fig, 'CurrentAxes', obj.axCam);
            cla(obj.axCam);
            hold(obj.axCam, 'on');
            axis(obj.axCam, 'equal');
            axis(obj.axCam, 'off');
            view(obj.axCam, 3);
            
            % Perspective projection for realistic view
            camproj(obj.axCam, 'perspective');
            
            % Compute camera orientation vectors
            [camTarget, camUp] = obj.calculateCameraVectors(cameraPos, cameraYaw);
            
            % Apply camera transforms
            campos(obj.axCam, cameraPos);
            camtarget(obj.axCam, camTarget);
            camup(obj.axCam, camUp);
            camva(obj.axCam, cfg.CAMERA_FOV_H);
            
            % Camera viewing bounds
            axisRange = cfg.CAMERA_RANGE * 1.2;
            xlim(obj.axCam, [cameraPos(1)-axisRange, cameraPos(1)+axisRange]);
            ylim(obj.axCam, [cameraPos(2)-axisRange, cameraPos(2)+axisRange]);
            zlim(obj.axCam, [0, cfg.ARENA_Z]);
            
            % Sky color background
            set(obj.axCam, 'Color', [0.5 0.6 0.8]);
            
            % ------------------ Ground -------------------------------
            patch(obj.axCam, ...
                [0 cfg.ARENA_X cfg.ARENA_X 0], ...
                [0 0 cfg.ARENA_Y cfg.ARENA_Y], ...
                [0 0 0 0], ...
                cfg.COLOR_PATHWAY, ...
                'FaceAlpha', 0.5, 'EdgeColor', 'none');
            
            % ------------------ Scene Geometry -----------------------
            Renderer.drawBlocksInCameraView(obj.axCam, arena.blocks, ...
                                            cameraPos, cfg);
            
            kfsInView = Renderer.drawKFSInCameraView(obj.axCam, arena, ...
                cameraPos, cameraYaw, vision.detectedKfs, cfg);
            
            % ------------------ HUD Metrics --------------------------
            detectedR2   = sum(navigator.belief == "R2",   'all');
            detectedR1   = sum(navigator.belief == "R1",   'all');
            detectedFake = sum(navigator.belief == "FAKE", 'all');
            
            % Detected counts overlay
            text(obj.axCam, 0.02, 0.98, sprintf( ...
                'DETECTED → R2: %d/%d | R1: %d/%d | FAKE: %d/%d | InView: %d', ...
                detectedR2, cfg.TOTAL_R2_REAL, ...
                detectedR1, cfg.TOTAL_R1, ...
                detectedFake, 1, ...
                kfsInView), ...
                'Units', 'normalized', 'Color', [0 1 0], ...
                'FontSize', 11, 'FontWeight', 'bold', ...
                'BackgroundColor', [0 0 0 0.7], ...
                'VerticalAlignment', 'top');
            
            % Carry / mode overlay
            text(obj.axCam, 0.02, 0.93, sprintf( ...
                'PICKED → Carry: %d/%d | Mode: %s', ...
                navigator.carry, cfg.CAPACITY, navigator.mode), ...
                'Units', 'normalized', 'Color', [1 1 0], ...
                'FontSize', 11, 'FontWeight', 'bold', ...
                'BackgroundColor', [0 0 0 0.7], ...
                'VerticalAlignment', 'top');
            
            % Camera telemetry overlay
            text(obj.axCam, 0.02, 0.86, sprintf( ...
                'Pitch: %.0f° | Yaw: %.0f° | Auto: %d', ...
                obj.cameraPitch, cameraYaw, obj.autonomousMode), ...
                'Units', 'normalized', 'Color', 'white', ...
                'FontSize', 10, 'BackgroundColor', [0 0 0 0.7], ...
                'VerticalAlignment', 'top');
            
            % Recording indicator
            if robot.recordingPath
                text(obj.axCam, 0.98, 0.98, '● REC', ...
                    'Units', 'normalized', 'Color', 'red', ...
                    'FontSize', 14, 'FontWeight', 'bold', ...
                    'HorizontalAlignment', 'right', ...
                    'BackgroundColor', [0 0 0 0.7], ...
                    'VerticalAlignment', 'top');
            end
            
            % Crosshair overlay
            text(obj.axCam, 0.5, 0.5, '+', ...
                'Units', 'normalized', 'Color', [0 1 0], ...
                'FontSize', 24, 'HorizontalAlignment', 'center', ...
                'VerticalAlignment', 'middle');
            
            % Panel title
            title(obj.axCam, 'CAMERA FEED (Vision+Nav Active)', ...
                  'Color', 'white', 'FontSize', 14);
        end
        
        
        % ===================================================================
        % Camera Vector Calculation
        % -------------------------------------------------------------------
        % Computes camera target and up vectors from yaw and pitch.
        % ===================================================================
        function [camTarget, camUp] = calculateCameraVectors(obj, cameraPos, cameraYaw)
            
            % Convert angles from degrees to radians
            yawRad   = deg2rad(cameraYaw);
            pitchRad = deg2rad(obj.cameraPitch);
            
            % Rotation matrix for yaw (rotation about Z axis)
            R_yaw = [ cos(yawRad) -sin(yawRad) 0; ...
                      sin(yawRad)  cos(yawRad) 0; ...
                      0            0           1 ];
            
            % Rotation matrix for pitch (rotation about X axis)
            R_pitch = [ 1  0               0; ...
                        0  cos(pitchRad) -sin(pitchRad); ...
                        0  sin(pitchRad)  cos(pitchRad) ];
            
            % Combined camera rotation
            R_cam = R_yaw * R_pitch;
            
            % Forward projection vector defines camera target
            camTarget = cameraPos + (R_cam * [0; -1000; 0])';
            
            % Up vector defines camera roll orientation
            camUp = (R_cam * [0; 0; 1])';
        end
        
        
        % ===================================================================
        % Automatic Camera Pitch Adjustment
        % -------------------------------------------------------------------
        % Dynamically tracks the nearest visible KFS vertically.
        % ===================================================================
        function pitch = autoAdjustPitch(obj, robotState, arena, detections)
            
            % Persistent target allows smooth convergence across frames
            persistent targetPitch;
            if isempty(targetPitch), targetPitch = 15; end
            
            cfg = obj.config;
            
            % Camera pose
            cameraPos = robotState.position + ...
                        [0 0 cfg.CAMERA_HEIGHT_OFFSET];
            
            cameraYaw = robotState.yaw + obj.cameraYawOffset;
            
            % Forward direction vector projected onto XY plane
            camFwd = [cos(deg2rad(cameraYaw-90)), ...
                      sin(deg2rad(cameraYaw-90)), 0];
            
            minDist = inf;
            bestKfs = [];
            
            % Iterate over all KFS blocks
            for i = 1:numel(arena.kfsIds)
                
                kb = arena.blocks(arena.kfsIds(i));
                kfsCenter = [kb.x, kb.y, ...
                             kb.h + cfg.KFS_SIZE/2];
                
                toKfs = kfsCenter - cameraPos;
                dist  = norm(toKfs);
                
                % Ignore distant targets
                if dist > cfg.DETECTION_RANGE
                    continue;
                end
                
                % Only consider objects in front of the camera
                if dot(toKfs(1:2), camFwd(1:2)) > 0 && dist < minDist
                    minDist = dist;
                    bestKfs = kfsCenter;
                end
            end
            
            % Compute desired pitch angle to target
            if ~isempty(bestKfs)
                deltaZ = bestKfs(3) - cameraPos(3);
                horizontalDist = norm(bestKfs(1:2) - cameraPos(1:2));
                targetPitch = atand(deltaZ / horizontalDist);
                
                % Clamp pitch to safe limits
                targetPitch = max(-45, min(45, targetPitch));
            end
            
            % Smooth interpolation toward target pitch
            pitch = obj.cameraPitch + ...
                    (targetPitch - obj.cameraPitch) * 0.1;
        end
        
        
        % ===================================================================
        % Help Overlay Renderer
        % -------------------------------------------------------------------
        % Displays keyboard controls in the overview panel.
        % ===================================================================
        function renderHelpText(obj)
            
            axes(obj.axMain);
            
            helpText = sprintf([ ...
                ' CONTROLS:\n' ...
                ' W/S  : Fwd/Back\n' ...
                ' A/D  : Strafe\n' ...
                ' Q/E  : Rotate\n' ...
                ' I/K  : Pan Camera\n' ...
                ' Z/X  : Pitch\n' ...
                ' V    : Auto-Tilt\n' ...
                ' N    : Toggle NAV\n' ...
                ' SPC  : Climb\n' ...
                ' C    : Down\n' ...
                ' R    : Reset\n' ...
                ' T    : Randomize\n' ...
                ' P    : Record\n' ...
                ' L    : Export\n' ...
                ' H    : Toggle Help']);
            
            text(0.02, 0.98, helpText, ...
                'Units', 'normalized', ...
                'VerticalAlignment', 'top', ...
                'Color', 'yellow', ...
                'BackgroundColor', [0 0 0 0.8], ...
                'FontName', 'FixedWidth', ...
                'FontSize', 9);
        end
        
        
        % ===================================================================
        % Input Processing Stub
        % -------------------------------------------------------------------
        % Input is handled asynchronously via callbacks.
        % ===================================================================
        function processInput(obj, robot, arena, navigator)
            % Input processing is handled by callbacks
        end
        
        
        % ===================================================================
        % Runtime Status Accessors
        % ===================================================================
        function r = isRunning(obj)
            % Returns whether the simulation should continue running
            r = obj.running && ishandle(obj.fig);
        end
        
        function m = isAutonomousMode(obj)
            % Returns current autonomous navigation state
            m = obj.autonomousMode;
        end
    end
end
