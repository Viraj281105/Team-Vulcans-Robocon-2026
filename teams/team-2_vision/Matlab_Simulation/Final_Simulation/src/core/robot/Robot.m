classdef Robot < handle
    % ===================================================================
    % ROBOT
    % -------------------------------------------------------------------
    % Manages robot physical state, movement kinematics, collision rules,
    % grid alignment, camera orientation state, and telemetry tracking.
    %
    % Architectural role:
    %   - Acts as the authoritative source of robot pose and motion state.
    %   - Encapsulates all movement constraints and collision logic.
    %   - Maintains cumulative statistics and trajectory history.
    %   - Provides safe APIs for both manual and autonomous control.
    %
    % This class deliberately centralizes:
    %   - Position updates
    %   - Orientation updates
    %   - Height transitions (climb / descend)
    %   - Distance accumulation
    %   - Path recording
    %
    % This ensures deterministic behavior and eliminates state divergence.
    % ===================================================================
    
    properties
        % ------------------ Configuration Reference ---------------------
        config          % SimConfig object
        
        % ------------------ 3D Pose ------------------------------
        position        % [x, y, z] position in millimeters
        yaw             % Heading angle in degrees
        
        % ------------------ Camera Orientation State -------------------
        cameraYaw       % Camera yaw angle (degrees)
        cameraPitch     % Camera pitch angle (degrees)
        
        % ------------------ Grid Localization --------------------------
        currentBlockId  % ID of block currently occupied
        gridPos         % [row, col] grid coordinates
        
        % ------------------ Trajectory Recording -----------------------
        pathHistory     % Nx3 matrix of historical positions
        recordingPath   % Boolean flag enabling logging
        
        % ------------------ Distance Statistics ------------------------
        totalDistance   % Accumulated traveled distance (mm)
        lastPosition    % Last position snapshot for delta calculation
    end
    
    methods
        
        % ===================================================================
        % Constructor
        % -------------------------------------------------------------------
        % Initializes robot state from the arena's initial block.
        % ===================================================================
        function obj = Robot(config, initialBlock)
            
            % Store configuration reference
            obj.config = config;
            
            % Initialize block occupancy
            obj.currentBlockId = initialBlock.id;
            
            % Initialize robot position at block center on ground plane
            obj.position = [initialBlock.x, initialBlock.y, 0];
            
            % Initial yaw orientation (degrees)
            obj.yaw = 180;
            
            % ------------------ Camera Defaults --------------------------
            obj.cameraYaw   = 180;    % Camera aligned with robot heading
            obj.cameraPitch = 15;     % Slight downward tilt
            
            % Grid location initialization
            obj.gridPos = [initialBlock.row, initialBlock.col];
            
            % Trajectory recording defaults
            obj.pathHistory   = [];
            obj.recordingPath = false;
            
            % Distance tracking initialization
            obj.totalDistance = 0;
            obj.lastPosition  = obj.position;
        end
        
        
        % ===================================================================
        % State Snapshot Accessor
        % -------------------------------------------------------------------
        % Returns a lightweight structure of robot pose and grid state.
        % ===================================================================
        function state = getState(obj)
            
            state.position = obj.position;
            state.yaw       = obj.yaw;
            state.gridPos   = obj.gridPos;
            state.blockId   = obj.currentBlockId;
        end
        
        
        % ===================================================================
        % Convenience Movement Wrappers
        % -------------------------------------------------------------------
        % These forward commands into the unified move() primitive.
        % ===================================================================
        function [success, newPos] = moveForward(obj, speed, arena)
            [success, newPos] = obj.move(speed, 0, arena);
        end
        
        function [success, newPos] = moveBackward(obj, speed, arena)
            [success, newPos] = obj.move(-speed, 0, arena);
        end
        
        function [success, newPos] = strafeLeft(obj, speed, arena)
            [success, newPos] = obj.move(0, speed, arena);
        end
        
        function [success, newPos] = strafeRight(obj, speed, arena)
            [success, newPos] = obj.move(0, -speed, arena);
        end
        
        
        % ===================================================================
        % Unified Motion Primitive
        % -------------------------------------------------------------------
        % Applies forward and lateral motion relative to robot heading.
        % Performs boundary checks and height logic.
        % ===================================================================
        function [success, newPos] = move(obj, forward, side, arena)
            
            % Convert yaw to radians.
            % Subtracting 90° aligns heading with MATLAB coordinate frame.
            yawRad = deg2rad(obj.yaw - 90);
            
            % Resolve forward and lateral motion into world frame deltas
            dx = forward * cos(yawRad) - side * sin(yawRad);
            dy = forward * sin(yawRad) + side * cos(yawRad);
            
            % Proposed new position (Z unchanged initially)
            newPos = obj.position + [dx, dy, 0];
            
            % ---------------------------------------------------------------
            % Arena boundary collision check
            % ---------------------------------------------------------------
            cfg = obj.config;
            
            if newPos(1) < cfg.R2_WIDTH/2 || ...
               newPos(1) > cfg.ARENA_X - cfg.R2_WIDTH/2 || ...
               newPos(2) < cfg.R2_LENGTH/2 || ...
               newPos(2) > cfg.ARENA_Y - cfg.R2_LENGTH/2
                
                success = false;
                fprintf('[COLLISION] Arena boundary\n');
                return;
            end
            
            % ---------------------------------------------------------------
            % Determine which block the robot is over
            % ---------------------------------------------------------------
            onBlockId = 0;
            
            for k = 1:numel(arena.blocks)
                b = arena.blocks(k);
                
                if abs(newPos(1) - b.x) < cfg.BLOCK_SIZE/2 && ...
                   abs(newPos(2) - b.y) < cfg.BLOCK_SIZE/2
                    onBlockId = b.id;
                    break;
                end
            end
            
            % ---------------------------------------------------------------
            % Height-based collision and placement logic
            % ---------------------------------------------------------------
            if obj.position(3) <= 10        % Robot is on ground plane
                
                newPos(3) = 0;
                
                % Prevent exiting forest bounds while on ground
                if ~arena.isInForestBounds(newPos(1:2))
                    success = false;
                    fprintf('[COLLISION] Off forest area (ground)\n');
                    return;
                end
                
                % Update block occupancy if standing on block
                if onBlockId ~= 0
                    obj.currentBlockId = onBlockId;
                    b = arena.blocks(onBlockId);
                    obj.gridPos = [b.row, b.col];
                end
                
                success = true;
                
            else                            % Robot is elevated on block
                
                % Prevent exiting forest while elevated
                if ~arena.isInForestBounds(newPos(1:2))
                    success = false;
                    fprintf('[COLLISION] Off blocks (elevated)\n');
                    return;
                end
                
                % Maintain elevation unless stepping onto a block
                newPos(3) = obj.position(3);
                
                if onBlockId ~= 0
                    b = arena.blocks(onBlockId);
                    newPos(3) = b.h;
                    obj.currentBlockId = b.id;
                    obj.gridPos = [b.row, b.col];
                end
                
                success = true;
            end
            
            % ---------------------------------------------------------------
            % Commit motion if successful
            % ---------------------------------------------------------------
            if success
                obj.position = newPos;
                obj.updateStats();
                obj.recordPath();
            end
        end
        
        
        % ===================================================================
        % Rotation Control
        % -------------------------------------------------------------------
        % Adjusts robot yaw angle.
        % ===================================================================
        function rotate(obj, deltaYaw)
            obj.yaw = obj.yaw + deltaYaw;
        end
        
        
        % ===================================================================
        % Vertical Motion: Climb
        % -------------------------------------------------------------------
        % Raises robot onto the top surface of the block beneath it.
        % ===================================================================
        function climbToBlock(obj, arena)
            
            cfg = obj.config;
            
            for k = 1:numel(arena.blocks)
                b = arena.blocks(k);
                
                if abs(obj.position(1) - b.x) < cfg.BLOCK_SIZE/2 && ...
                   abs(obj.position(2) - b.y) < cfg.BLOCK_SIZE/2
                    
                    obj.position(3)     = b.h;
                    obj.currentBlockId  = b.id;
                    obj.gridPos         = [b.row, b.col];
                    
                    fprintf('[CLIMB] Block %d (H=%d)\n', b.id, b.h);
                    return;
                end
            end
        end
        
        
        % ===================================================================
        % Vertical Motion: Descend
        % -------------------------------------------------------------------
        % Returns robot to ground plane.
        % ===================================================================
        function descend(obj)
            obj.position(3) = 0;
            fprintf('[DESCEND] Ground level\n');
        end
        
        
        % ===================================================================
        % Direct Position Assignment (Autonomous Control)
        % -------------------------------------------------------------------
        % Used by planner to teleport robot within constraints.
        % ===================================================================
        function moveTo(obj, newPos, arena)
            
            % Update horizontal position directly
            obj.position(1:2) = newPos(1:2);
            
            % Determine which block robot is over
            cfg = obj.config;
            
            for k = 1:numel(arena.blocks)
                b = arena.blocks(k);
                
                if abs(obj.position(1) - b.x) < cfg.BLOCK_SIZE/2 && ...
                   abs(obj.position(2) - b.y) < cfg.BLOCK_SIZE/2
                    
                    obj.currentBlockId = b.id;
                    obj.gridPos        = [b.row, b.col];
                    
                    % Maintain elevation consistency
                    if obj.position(3) > 10
                        obj.position(3) = b.h;
                    end
                    break;
                end
            end
            
            % Update telemetry
            obj.updateStats();
            obj.recordPath();
        end
        
        
        % ===================================================================
        % Path Recording Control
        % -------------------------------------------------------------------
        % Enables or disables trajectory logging.
        % ===================================================================
        function setRecording(obj, enabled)
            obj.recordingPath = enabled;
        end
        
        
        % ===================================================================
        % Reset Robot State
        % -------------------------------------------------------------------
        % Restores robot to initial arena position and clears telemetry.
        % ===================================================================
        function reset(obj, arena)
            
            initialBlock = arena.getInitialBlock();
            
            obj.currentBlockId = initialBlock.id;
            obj.position       = [initialBlock.x, initialBlock.y, 0];
            
            obj.yaw         = 0;
            obj.cameraYaw   = 0;
            obj.cameraPitch = 15;
            
            obj.gridPos     = [initialBlock.row, initialBlock.col];
            obj.pathHistory = [];
            
            obj.totalDistance = 0;
            obj.lastPosition  = obj.position;
            
            fprintf('[RESET] Robot reset\n');
        end
        
        
        % ===================================================================
        % Distance Accumulator
        % -------------------------------------------------------------------
        % Updates total traveled distance incrementally.
        % ===================================================================
        function updateStats(obj)
            obj.totalDistance = ...
                obj.totalDistance + norm(obj.position - obj.lastPosition);
            obj.lastPosition = obj.position;
        end
        
        
        % ===================================================================
        % Trajectory Recorder
        % -------------------------------------------------------------------
        % Appends current position to history if recording enabled.
        % ===================================================================
        function recordPath(obj)
            if obj.recordingPath
                obj.pathHistory(end+1, :) = obj.position;
            end
        end
        
        
        % ===================================================================
        % Distance Accessor
        % -------------------------------------------------------------------
        % Returns cumulative traveled distance.
        % ===================================================================
        function dist = getTotalDistance(obj)
            dist = obj.totalDistance;
        end
    end
end
