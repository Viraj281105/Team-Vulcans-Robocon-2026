function robocon_arena_sim()
% =======================================================================
% ROBOCON 2026 – 3D SIMULATION + TIME-AWARE HYBRID A* NAVIGATION
% -----------------------------------------------------------------------
% This function is the MAIN ENTRY POINT of the entire simulator.
%
% Responsibilities of this file:
%   - Reset and clean the MATLAB graphical environment.
%   - Instantiate all core subsystems:
%         • Simulation configuration
%         • Arena and environment
%         • Robot model
%         • Vision perception pipeline
%         • Navigation / planning engine
%         • User interface and input handler
%   - Wire dependencies between subsystems.
%   - Execute the real-time simulation loop.
%   - Coordinate vision updates, belief propagation, navigation logic,
%     and rendering every frame.
%
% Architecturally, this function behaves like:
%   → A runtime orchestrator / system controller.
%   → It does NOT implement domain logic directly.
%   → It simply coordinates and synchronizes independent modules.
%
% This separation ensures:
%   - Modularity
%   - Maintainability
%   - Easy debugging and extension
%
% =======================================================================
% Controls: Press 'H' in simulation for help
% =======================================================================


%% ========================= ENVIRONMENT CLEANUP =========================
% The goal of this section is to guarantee a clean MATLAB state before the
% simulation starts. This avoids:
%   - Stale figure handles
%   - Orphaned UI windows
%   - Residual graphical objects
%   - Conflicting callbacks
%
% This is especially important for interactive simulations where leftover
% figures or callbacks can silently break input handling or rendering.


% Enable visibility of hidden graphics handles.
% This allows MATLAB to access and delete ALL graphical objects, including
% internally managed UI figures that might otherwise persist.
set(groot, 'ShowHiddenHandles', 'on');

% Delete all children of the root graphics object.
% Effectively closes all open figures, axes, UI windows, and graphics handles.
delete(get(groot, 'Children'));

% Clear the command window for clean logs and readable output.
clc;


%% ========================= CONFIGURATION SETUP ==========================
% Create the simulation configuration object.
%
% The SimConfig class typically encapsulates:
%   - Arena dimensions
%   - Robot physical parameters
%   - Camera properties (FOV, mounting offset, pitch limits, etc.)
%   - Navigation grid resolution
%   - Timing constants
%   - Rendering preferences
%
% This acts as a SINGLE SOURCE OF TRUTH for all subsystems, ensuring that
% every module operates with consistent assumptions.
config = SimConfig();


%% ========================= ARENA INITIALIZATION =========================
% Create the Arena object using the shared configuration.
%
% The Arena object usually:
%   - Builds the 3D environment geometry
%   - Generates forest blocks, obstacles, and terrain
%   - Stores spatial layout and occupancy data
%   - Provides helper APIs for collision queries and visualization
arena = Arena(config);


%% ========================= ROBOT INITIALIZATION =========================
% Create the Robot object.
%
% Inputs:
%   - config → Provides robot kinematics, size, sensor offsets, etc.
%   - arena.getInitialBlock() → Specifies the robot's spawn location
%
% The Robot object maintains:
%   - Position and orientation
%   - Camera pose (yaw / pitch)
%   - Motion constraints
%   - Interaction with terrain
robot = Robot(config, arena.getInitialBlock());


%% ========================= VISION SYSTEM INITIALIZATION =================
% Instantiate the vision pipeline.
%
% The VisionSystem typically:
%   - Simulates camera sensing
%   - Performs visibility checks
%   - Detects KFS blocks or objects
%   - Maintains probabilistic belief grids
%   - Models occlusion and noise
%
% It depends on:
%   - config → Camera parameters and thresholds
%   - arena  → Geometry for ray casting and visibility
vision = VisionSystem(config, arena);


%% ========================= NAVIGATION SYSTEM INITIALIZATION ==============
% Instantiate the navigation / planning engine.
%
% The Navigator typically:
%   - Maintains belief maps of the environment
%   - Executes Hybrid A* or grid-based planning
%   - Handles mission logic (collection, exit, completion)
%   - Issues motion commands to the robot
%
% It depends on:
%   - config → Grid resolution, planner constraints
%   - arena  → Traversability and obstacles
navigator = Navigator(config, arena);


%% ========================= USER INTERFACE INITIALIZATION =================
% Create the Simulation UI.
%
% The UI is responsible for:
%   - Rendering the arena and robot
%   - Displaying HUD overlays
%   - Capturing keyboard input
%   - Providing runtime toggles (autonomous mode, camera controls, etc.)
%
% UI is intentionally separated from logic for clean architecture.
ui = SimulationUI(config, arena);


%% ========================= INPUT HANDLER WIRING ==========================
% Provide references of all major subsystems to the input handler.
%
% This allows keyboard events to:
%   - Control robot movement
%   - Toggle navigation modes
%   - Trigger vision modes
%   - Interact with arena state
%
% Dependency injection avoids global variables and keeps coupling explicit.
ui.inputHandler.setReferences(robot, arena, navigator, vision);


%% ========================= SIMULATION START BANNER =======================
% Print startup information to the console.
%
% This provides:
%   - User feedback that the simulation is running
%   - Reminder of key controls
%   - Debug visibility for logs
fprintf('=== ROBOCON 2026 - 3D + NAV Simulation ===\n');
fprintf('Press "V" for auto-tilt, "N" to toggle autonomous navigation\n');
fprintf('Press "H" for help\n');


%% ========================= MAIN SIMULATION LOOP ==========================
% This loop runs continuously until the UI signals termination.
%
% High-level loop responsibilities per frame:
%   1. Acquire sensor data (vision)
%   2. Update belief state
%   3. Execute navigation logic (if autonomous)
%   4. Render updated state
%   5. Process UI events
%
% This loop effectively acts as a real-time control loop.
while ui.isRunning()
    
    % ========================= VISION PROCESSING =========================
    % Process the current frame from the robot's perspective.
    %
    % Inputs:
    %   - robot.getState() → Current robot pose and camera orientation
    %   - arena             → Environment geometry for visibility
    %
    % Output:
    %   - detections → Detected objects / KFS / features in view
    detections = vision.processFrame(robot.getState(), arena);
    
    
    % ========================= BELIEF UPDATE (CRITICAL FIX) ================
    % Always synchronize the navigator's belief state from vision.
    %
    % Why this matters:
    %   - Ensures HUD reflects real-time detection counts.
    %   - Keeps planning state consistent with perception.
    %   - Prevents stale belief accumulation.
    %
    % This explicitly forces the navigator to trust vision updates every frame.
    navigator.updateBeliefFromDetections(vision, arena);
    
    
    % ========================= CAMERA POSE EXTRACTION ======================
    % Retrieve the robot's current state structure.
    robotState = robot.getState();
    
    % Compute absolute camera position in world coordinates.
    % The camera is vertically offset above the robot base by a fixed amount.
    cameraPos = robotState.position + [0 0 config.CAMERA_HEIGHT_OFFSET];
    
    % Extract camera yaw (horizontal rotation).
    cameraYaw = robot.cameraYaw;
    
    % Extract camera pitch (vertical tilt).
    cameraPitch = robot.cameraPitch;
    
    
    % ========================= GRID BELIEF UPDATE ==========================
    % Update the navigator's probabilistic grid belief using camera geometry.
    %
    % Inputs:
    %   - cameraPos           → Camera world position
    %   - cameraYaw           → Heading direction
    %   - cameraPitch         → Vertical tilt
    %   - navigator.belief    → Current belief grid
    %   - navigator.beliefConfidence → Confidence values per cell
    %
    % Outputs:
    %   - Updated belief grid
    %   - Updated confidence grid
    %
    % This models visibility, occlusion, and sensing uncertainty.
    [navigator.belief, navigator.beliefConfidence] = vision.updateGridBelief(...
        cameraPos, cameraYaw, cameraPitch, navigator.belief, navigator.beliefConfidence);
    
    
    % ========================= AUTONOMOUS NAVIGATION =======================
    % If autonomous mode is enabled by the user:
    %   - Execute one navigation step
    %   - Issue movement commands to the robot
    %   - Evaluate mission completion
    if ui.isAutonomousMode()
        
        % Perform one planning + execution step.
        navigator.step(robot, arena, vision, detections);
        
        % Check whether the mission objectives are complete.
        if navigator.missionComplete
            
            % Display success message with performance metrics.
            fprintf('\n🎉 MISSION SUCCESS! Autonomous navigation complete.\n');
            fprintf('   Collected: %d items | Time: %.1fs\n\n', ...
                navigator.carry, toc(navigator.gameStartTime));
            
            % Disable autonomous mode after completion.
            ui.autonomousMode = false;
            
            % Reset mission flag for potential next run.
            navigator.missionComplete = false;
        end
    end
    
    
    % ========================= RENDERING ================================
    % Render the updated simulation state.
    %
    % The UI typically:
    %   - Draws arena geometry
    %   - Draws robot pose and trajectory
    %   - Displays detections and belief maps
    %   - Updates HUD overlays
    ui.render(robot, arena, vision, navigator, detections);
    
    
    % Force MATLAB to flush graphics events immediately.
    % This keeps the animation responsive and interactive.
    drawnow;
end


%% ========================= SHUTDOWN MESSAGE ==============================
% Print termination message when the simulation loop exits.
fprintf('Simulation ended\n');

end
