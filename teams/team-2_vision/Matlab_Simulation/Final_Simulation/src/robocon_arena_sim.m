function robocon_arena_sim()
% =======================================================================
% ROBOCON 2026 – 3D SIMULATION + TIME-AWARE HYBRID A* NAVIGATION
% Main entry point for the simulation
% =======================================================================
% Controls: Press 'H' in simulation for help
% =======================================================================

%% Clean environment
set(groot, 'ShowHiddenHandles', 'on');
delete(get(groot, 'Children'));
clc;

%% Initialize configuration
config = SimConfig();

%% Initialize arena and forest
arena = Arena(config);

%% Initialize robot
robot = Robot(config, arena.getInitialBlock());

%% Initialize vision system
vision = VisionSystem(config, arena);

%% Initialize navigation system
navigator = Navigator(config, arena);

%% Initialize UI
ui = SimulationUI(config, arena);

%% Wire up input handler with all components
ui.inputHandler.setReferences(robot, arena, navigator, vision);

%% Main simulation loop
fprintf('=== ROBOCON 2026 - 3D + NAV Simulation ===\n');
fprintf('Press "V" for auto-tilt, "N" to toggle autonomous navigation\n');
fprintf('Press "H" for help\n');

while ui.isRunning()
    % Update vision detections
    detections = vision.processFrame(robot.getState(), arena);
    
    % Update navigation if autonomous
    if ui.isAutonomousMode()
        navigator.step(robot, arena, vision, detections);
        
        % Check if mission completed
        if navigator.missionComplete
            fprintf('\n🎉 MISSION SUCCESS! Autonomous navigation complete.\n');
            fprintf('   Collected: %d items | Time: %.1fs\n\n', ...
                navigator.carry, toc(navigator.gameStartTime));
            ui.autonomousMode = false;  % Turn off autonomous mode
            navigator.missionComplete = false;  % Reset for next run
        end
    end
    
    % Update UI
    ui.render(robot, arena, vision, navigator, detections);
    
    drawnow;
end

fprintf('Simulation ended\n');

end
