classdef SimulationUI < handle
    % SIMULATIONUI Manages the graphical interface and rendering
    
    properties
        config
        arena
        
        % Figure and axes
        fig
        axMain
        axCam
        
        % UI state
        running
        showHelp
        autoTiltEnabled
        autonomousMode
        
        % Camera state
        cameraPitch
        cameraYawOffset
        
        % Input handler
        inputHandler
    end
    
    methods
        function obj = SimulationUI(config, arena)
            obj.config = config;
            obj.arena = arena;
            
            obj.running = true;
            obj.showHelp = false;
            obj.autoTiltEnabled = false;
            obj.autonomousMode = false;
            
            obj.cameraPitch = 15;
            obj.cameraYawOffset = 0;
            
            obj.setupFigure();
            obj.inputHandler = InputHandler(obj);
            
            % CRITICAL FIX: Ensure figure stays in focus
            fprintf('[UI] Keyboard controls enabled - figure must have focus\n');
        end
        
        function setupFigure(obj)
            obj.fig = figure('Name', 'ROBOCON 2026 - 3D + NAV Simulation', ...
                'Position', [50 50 1800 900], ...
                'Color', [0.15 0.15 0.15], ...
                'NumberTitle', 'off', ...
                'MenuBar', 'none', ...
                'ToolBar', 'figure', ...
                'WindowKeyPressFcn', @(src, event) obj.handleKeyPress(src, event));  % FIXED: Use WindowKeyPressFcn
            
            obj.axMain = subplot(2, 3, [1 4]);
            obj.axCam = subplot(2, 3, [2 3 5 6]);
            
            % CRITICAL FIX: Make figure interruptible for key events
            set(obj.fig, 'BusyAction', 'cancel', 'Interruptible', 'on');
        end
        
        function handleKeyPress(obj, ~, event)
            % Filter out system/modifier keys
            systemKeys = {'alt', 'control', 'shift', 'tab', 'windows', 'command', 'capslock'};
            
            if ismember(lower(event.Key), systemKeys)
                return;  % Silently ignore system keys - this allows Alt+Tab!
            end
            
            fprintf('[UI DEBUG] Key pressed: %s\n', event.Key);
            
            % Handle UI-only keys directly (don't pass to InputHandler)
            switch event.Key
                case 'n'
                    obj.autonomousMode = ~obj.autonomousMode;
                    fprintf('[NAV] Autonomous mode %d\n', obj.autonomousMode);
                    return;  % CRITICAL: Return here to prevent double-toggle!
                    
                case 'h'
                    obj.showHelp = ~obj.showHelp;
                    fprintf('[UI] Help overlay %d\n', obj.showHelp);
                    return;  % Don't pass to InputHandler
                    
                case 'v'
                    obj.autoTiltEnabled = ~obj.autoTiltEnabled;
                    fprintf('[CAMERA] Auto-tilt %d\n', obj.autoTiltEnabled);
                    return;  % Don't pass to InputHandler
                    
                case 'escape'
                    obj.running = false;
                    fprintf('[UI] Simulation ending...\n');
                    close(obj.fig);
                    return;
            end
            
            % ONLY pass robot control keys (WASD, QE, etc.) to InputHandler
            % Don't pass UI toggle keys
            if ~isempty(obj.inputHandler)
                obj.inputHandler.handleKey([], event);
            end
        end


        
        function render(obj, robot, arena, vision, navigator, detections)
            % Main render function
            if ~obj.running || ~ishandle(obj.fig)
                return;
            end
            
            % CRITICAL FIX: Ensure figure stays in focus
            if ~strcmp(get(obj.fig, 'CurrentObject'), '')
                figure(obj.fig);  % Bring figure to front if it lost focus
            end
            
            robotState = robot.getState();
            
            % Update camera if auto-tilt enabled
            if obj.autoTiltEnabled
                obj.cameraPitch = obj.autoAdjustPitch(robotState, arena, detections);
            end
            
            % Render overview (left panel)
            obj.renderOverview(robot, arena, vision, navigator);
            
            % Render camera view (right panel)
            obj.renderCameraView(robot, arena, vision, detections, navigator);
            
            % Force graphics update
            drawnow limitrate;
        end
        
        function renderOverview(obj, robot, arena, vision, navigator)
            % Render the 3D overview (left panel)
            cfg = obj.config;
            set(obj.fig, 'CurrentAxes', obj.axMain);
            cla(obj.axMain);
            hold(obj.axMain, 'on');
            grid(obj.axMain, 'on');
            axis(obj.axMain, 'equal');
            
            xlabel(obj.axMain, 'X (mm)');
            ylabel(obj.axMain, 'Y (mm)');
            zlabel(obj.axMain, 'Z (mm)');
            view(obj.axMain, 45, 30);
            xlim(obj.axMain, [0 cfg.ARENA_X]);
            ylim(obj.axMain, [0 cfg.ARENA_Y]);
            zlim(obj.axMain, [0 cfg.ARENA_Z]);
            set(obj.axMain, 'Color', [0.05 0.05 0.1]);
            
            % Draw ground
            patch(obj.axMain, [0 cfg.ARENA_X cfg.ARENA_X 0], ...
                [0 0 cfg.ARENA_Y cfg.ARENA_Y], [0 0 0 0], ...
                cfg.COLOR_PATHWAY, 'FaceAlpha', 0.3, 'EdgeColor', 'none');
            
            % Draw blocks
            Renderer.drawBlocks(obj.axMain, arena.blocks, robot.currentBlockId, cfg);
            
            % Draw KFS
            Renderer.drawKFS(obj.axMain, arena, vision.detectedKfs, cfg);
            
            % Draw robot
            Renderer.drawRobot(obj.axMain, robot.getState(), cfg);
            
            % Draw detection range
            Renderer.drawDetectionRange(obj.axMain, robot.position, cfg);
            
            % Draw path history
            if ~isempty(robot.pathHistory)
                plot3(obj.axMain, robot.pathHistory(:,1), robot.pathHistory(:,2), ...
                    robot.pathHistory(:,3), 'g-', 'LineWidth', 2);
            end
            
            % Title with stats
            title(obj.axMain, sprintf('Mission: %.1fs | Dist: %.1fm | Yaw: %.0f° | Carry: %d/%d | Auto: %d', ...
                toc(navigator.gameStartTime), robot.getTotalDistance()/1000, ...
                robot.yaw, navigator.carry, cfg.CAPACITY, obj.autonomousMode), ...
                'Color', 'white');
            
            % Help overlay
            if obj.showHelp
                obj.renderHelpText();
            end
        end
        
        function renderCameraView(obj, robot, arena, vision, detections, navigator)
            % Render camera view (right panel)
            cfg = obj.config;
            robotState = robot.getState();
            
            cameraPos = robotState.position + [0 0 cfg.CAMERA_HEIGHT_OFFSET];
            cameraYaw = robotState.yaw + obj.cameraYawOffset;
            
            set(obj.fig, 'CurrentAxes', obj.axCam);
            cla(obj.axCam);
            hold(obj.axCam, 'on');
            axis(obj.axCam, 'equal');
            axis(obj.axCam, 'off');
            view(obj.axCam, 3);
            
            camproj(obj.axCam, 'perspective');
            
            % Setup camera
            [camTarget, camUp] = obj.calculateCameraVectors(cameraPos, cameraYaw);
            campos(obj.axCam, cameraPos);
            camtarget(obj.axCam, camTarget);
            camup(obj.axCam, camUp);
            camva(obj.axCam, cfg.CAMERA_FOV_H);
            
            axisRange = cfg.CAMERA_RANGE * 1.2;
            xlim(obj.axCam, [cameraPos(1)-axisRange, cameraPos(1)+axisRange]);
            ylim(obj.axCam, [cameraPos(2)-axisRange, cameraPos(2)+axisRange]);
            zlim(obj.axCam, [0, cfg.ARENA_Z]);
            
            set(obj.axCam, 'Color', [0.5 0.6 0.8]);
            
            % Draw scene
            patch(obj.axCam, [0 cfg.ARENA_X cfg.ARENA_X 0], ...
                [0 0 cfg.ARENA_Y cfg.ARENA_Y], [0 0 0 0], ...
                cfg.COLOR_PATHWAY, 'FaceAlpha', 0.5, 'EdgeColor', 'none');
            
            Renderer.drawBlocksInCameraView(obj.axCam, arena.blocks, cameraPos, cfg);
            kfsInView = Renderer.drawKFSInCameraView(obj.axCam, arena, ...
                cameraPos, cameraYaw, vision.detectedKfs, cfg);
            
            % HUD - Show detected vs picked counts
            detectedR2 = sum(navigator.belief == "R2", 'all');
            detectedR1 = sum(navigator.belief == "R1", 'all');
            detectedFake = sum(navigator.belief == "FAKE", 'all');
            
            % First line: Detected counts
            text(obj.axCam, 0.02, 0.98, sprintf('DETECTED → R2: %d/%d | R1: %d/%d | FAKE: %d/%d | InView: %d', ...
                detectedR2, cfg.TOTAL_R2_REAL, ...
                detectedR1, cfg.TOTAL_R1, ...
                detectedFake, 1, ...
                kfsInView), ...
                'Units', 'normalized', 'Color', [0 1 0], 'FontSize', 11, ...
                'FontWeight', 'bold', 'BackgroundColor', [0 0 0 0.7], ...
                'VerticalAlignment', 'top');
            
            % Second line: Picked/Carry count
            text(obj.axCam, 0.02, 0.93, sprintf('PICKED → Carry: %d/%d | Mode: %s', ...
                navigator.carry, cfg.CAPACITY, navigator.mode), ...
                'Units', 'normalized', 'Color', [1 1 0], 'FontSize', 11, ...
                'FontWeight', 'bold', 'BackgroundColor', [0 0 0 0.7], ...
                'VerticalAlignment', 'top');
            
            % Third line: Camera info
            text(obj.axCam, 0.02, 0.86, sprintf('Pitch: %.0f° | Yaw: %.0f° | Auto: %d', ...
                obj.cameraPitch, cameraYaw, obj.autonomousMode), ...
                'Units', 'normalized', 'Color', 'white', 'FontSize', 10, ...
                'BackgroundColor', [0 0 0 0.7], 'VerticalAlignment', 'top');
            
            % Recording indicator
            if robot.recordingPath
                text(obj.axCam, 0.98, 0.98, '● REC', 'Units', 'normalized', ...
                    'Color', 'red', 'FontSize', 14, 'FontWeight', 'bold', ...
                    'HorizontalAlignment', 'right', 'BackgroundColor', [0 0 0 0.7], ...
                    'VerticalAlignment', 'top');
            end
            
            % Crosshair
            text(obj.axCam, 0.5, 0.5, '+', 'Units', 'normalized', ...
                'Color', [0 1 0], 'FontSize', 24, 'HorizontalAlignment', 'center', ...
                'VerticalAlignment', 'middle');
            
            title(obj.axCam, 'CAMERA FEED (Vision+Nav Active)', 'Color', 'white', 'FontSize', 14);
        end
        
        function [camTarget, camUp] = calculateCameraVectors(obj, cameraPos, cameraYaw)
            yawRad = deg2rad(cameraYaw);
            pitchRad = deg2rad(obj.cameraPitch);
            
            R_yaw = [cos(yawRad) -sin(yawRad) 0; ...
                    sin(yawRad)  cos(yawRad) 0; ...
                    0            0           1];
            R_pitch = [1  0              0; ...
                      0  cos(pitchRad) -sin(pitchRad); ...
                      0  sin(pitchRad)  cos(pitchRad)];
            R_cam = R_yaw * R_pitch;
            
            camTarget = cameraPos + (R_cam * [0; -1000; 0])';
            camUp = (R_cam * [0; 0; 1])';
        end
        
        function pitch = autoAdjustPitch(obj, robotState, arena, detections)
            % Auto-adjust camera pitch to center on nearest KFS
            persistent targetPitch;
            if isempty(targetPitch), targetPitch = 15; end
            
            cfg = obj.config;
            cameraPos = robotState.position + [0 0 cfg.CAMERA_HEIGHT_OFFSET];
            cameraYaw = robotState.yaw + obj.cameraYawOffset;
            camFwd = [cos(deg2rad(cameraYaw-90)), sin(deg2rad(cameraYaw-90)), 0];
            
            minDist = inf;
            bestKfs = [];
            
            for i = 1:numel(arena.kfsIds)
                kb = arena.blocks(arena.kfsIds(i));
                kfsCenter = [kb.x, kb.y, kb.h + cfg.KFS_SIZE/2];
                
                toKfs = kfsCenter - cameraPos;
                dist = norm(toKfs);
                
                if dist > cfg.DETECTION_RANGE
                    continue;
                end
                
                if dot(toKfs(1:2), camFwd(1:2)) > 0 && dist < minDist
                    minDist = dist;
                    bestKfs = kfsCenter;
                end
            end
            
            if ~isempty(bestKfs)
                deltaZ = bestKfs(3) - cameraPos(3);
                horizontalDist = norm(bestKfs(1:2) - cameraPos(1:2));
                targetPitch = atand(deltaZ / horizontalDist);
                targetPitch = max(-45, min(45, targetPitch));
            end
            
            pitch = obj.cameraPitch + (targetPitch - obj.cameraPitch) * 0.1;
        end
        
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
            text(0.02, 0.98, helpText, 'Units', 'normalized', ...
                'VerticalAlignment', 'top', 'Color', 'yellow', ...
                'BackgroundColor', [0 0 0 0.8], 'FontName', 'FixedWidth', 'FontSize', 9);
        end
        
        function processInput(obj, robot, arena, navigator)
            % Input processing is handled by callbacks
        end
        
        function r = isRunning(obj)
            r = obj.running && ishandle(obj.fig);
        end
        
        function m = isAutonomousMode(obj)
            m = obj.autonomousMode;
        end
    end
end
