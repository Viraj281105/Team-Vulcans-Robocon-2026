classdef Robot < handle
    % ROBOT Manages robot state, position, and movement
    
    properties
        config          % SimConfig object
        
        % 3D position and orientation
        position        % [x, y, z] in mm
        yaw             % degrees
        
        % Grid state
        currentBlockId
        gridPos         % [row, col]
        
        % Path tracking
        pathHistory     % 3D position history
        recordingPath   % boolean
        
        % Stats
        totalDistance
        lastPosition
    end
    
    methods
        function obj = Robot(config, initialBlock)
            obj.config = config;
            obj.currentBlockId = initialBlock.id;
            obj.position = [initialBlock.x, initialBlock.y, 0];
            obj.yaw = 180;
            obj.gridPos = [initialBlock.row, initialBlock.col];
            obj.pathHistory = [];
            obj.recordingPath = false;
            obj.totalDistance = 0;
            obj.lastPosition = obj.position;
        end
        
        function state = getState(obj)
            % Return current robot state
            state.position = obj.position;
            state.yaw = obj.yaw;
            state.gridPos = obj.gridPos;
            state.blockId = obj.currentBlockId;
        end
        
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
        
        function [success, newPos] = move(obj, forward, side, arena)
            % Calculate new position based on movement commands
            yawRad = deg2rad(obj.yaw - 90);
            
            dx = forward * cos(yawRad) - side * sin(yawRad);
            dy = forward * sin(yawRad) + side * cos(yawRad);
            
            newPos = obj.position + [dx, dy, 0];
            
            % Check arena boundaries
            cfg = obj.config;
            if newPos(1) < cfg.R2_WIDTH/2 || newPos(1) > cfg.ARENA_X - cfg.R2_WIDTH/2 || ...
               newPos(2) < cfg.R2_LENGTH/2 || newPos(2) > cfg.ARENA_Y - cfg.R2_LENGTH/2
                success = false;
                fprintf('[COLLISION] Arena boundary\n');
                return;
            end
            
            % Find which block robot is over
            onBlockId = 0;
            for k = 1:numel(arena.blocks)
                b = arena.blocks(k);
                if abs(newPos(1) - b.x) < cfg.BLOCK_SIZE/2 && ...
                   abs(newPos(2) - b.y) < cfg.BLOCK_SIZE/2
                    onBlockId = b.id;
                    break;
                end
            end
            
            % Height-based collision detection
            if obj.position(3) <= 10  % On ground
                newPos(3) = 0;
                if ~arena.isInForestBounds(newPos(1:2))
                    success = false;
                    fprintf('[COLLISION] Off forest area (ground)\n');
                    return;
                end
                if onBlockId ~= 0
                    obj.currentBlockId = onBlockId;
                    b = arena.blocks(onBlockId);
                    obj.gridPos = [b.row, b.col];
                end
                success = true;
            else  % Elevated
                if ~arena.isInForestBounds(newPos(1:2))
                    success = false;
                    fprintf('[COLLISION] Off blocks (elevated)\n');
                    return;
                end
                newPos(3) = obj.position(3);
                if onBlockId ~= 0
                    b = arena.blocks(onBlockId);
                    newPos(3) = b.h;
                    obj.currentBlockId = b.id;
                    obj.gridPos = [b.row, b.col];
                end
                success = true;
            end
            
            if success
                obj.position = newPos;
                obj.updateStats();
                obj.recordPath();
            end
        end
        
        function rotate(obj, deltaYaw)
            obj.yaw = obj.yaw + deltaYaw;
        end
        
        function climbToBlock(obj, arena)
            % Climb to current block height
            cfg = obj.config;
            for k = 1:numel(arena.blocks)
                b = arena.blocks(k);
                if abs(obj.position(1) - b.x) < cfg.BLOCK_SIZE/2 && ...
                   abs(obj.position(2) - b.y) < cfg.BLOCK_SIZE/2
                    obj.position(3) = b.h;
                    obj.currentBlockId = b.id;
                    obj.gridPos = [b.row, b.col];
                    fprintf('[CLIMB] Block %d (H=%d)\n', b.id, b.h);
                    return;
                end
            end
        end
        
        function descend(obj)
            obj.position(3) = 0;
            fprintf('[DESCEND] Ground level\n');
        end
        
        function moveTo(obj, newPos, arena)
            % Direct position update (used by autonomous navigation)
            obj.position(1:2) = newPos(1:2);
            
            % Find block at new position (works for ground or elevated)
            cfg = obj.config;
            for k = 1:numel(arena.blocks)
                b = arena.blocks(k);
                if abs(obj.position(1) - b.x) < cfg.BLOCK_SIZE/2 && ...
                   abs(obj.position(2) - b.y) < cfg.BLOCK_SIZE/2
                    obj.currentBlockId = b.id;
                    obj.gridPos = [b.row, b.col];
                    
                    % Update height if elevated
                    if obj.position(3) > 10
                        obj.position(3) = b.h;
                    end
                    break;
                end
            end
            
            obj.updateStats();
            obj.recordPath();
        end
        
        function setRecording(obj, enabled)
            obj.recordingPath = enabled;
        end
        
        function reset(obj, arena)
            initialBlock = arena.getInitialBlock();
            obj.currentBlockId = initialBlock.id;
            obj.position = [initialBlock.x, initialBlock.y, 0];
            obj.yaw = 0;
            obj.gridPos = [initialBlock.row, initialBlock.col];
            obj.pathHistory = [];
            obj.totalDistance = 0;
            obj.lastPosition = obj.position;
            fprintf('[RESET] Robot reset\n');
        end
        
        function updateStats(obj)
            obj.totalDistance = obj.totalDistance + norm(obj.position - obj.lastPosition);
            obj.lastPosition = obj.position;
        end
        
        function recordPath(obj)
            if obj.recordingPath
                obj.pathHistory(end+1, :) = obj.position;
            end
        end
        
        function dist = getTotalDistance(obj)
            dist = obj.totalDistance;
        end
    end
end
