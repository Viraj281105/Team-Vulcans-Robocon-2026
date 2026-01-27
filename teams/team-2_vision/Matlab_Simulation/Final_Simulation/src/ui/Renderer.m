classdef Renderer
    % ===================================================================
    % RENDERER
    % -------------------------------------------------------------------
    % Static utility class responsible for ALL 3D drawing operations.
    %
    % Architectural role:
    %   - Provides reusable rendering primitives and scene drawing helpers.
    %   - Contains no simulation state.
    %   - Contains no side effects beyond graphics rendering.
    %   - Designed as a stateless functional service layer.
    %
    % Major responsibilities:
    %   - Drawing arena blocks with height-based coloring.
    %   - Drawing KFS cubes with optional textures.
    %   - Rendering the robot body and heading direction.
    %   - Drawing sensor range overlays.
    %   - Rendering camera-view geometry with visibility logic.
    %   - Generating 3D box meshes procedurally.
    %   - Projecting texture images onto cube faces.
    %
    % This keeps all visualization logic centralized and decoupled from:
    %   - UI event handling
    %   - Navigation logic
    %   - Physics / kinematics
    % ===================================================================
    
    methods (Static)
        
        % ===================================================================
        % Draw Arena Blocks (Overview View)
        % -------------------------------------------------------------------
        % Renders all forest blocks in the arena overview panel.
        % ===================================================================
        function drawBlocks(ax, blocks, currentBlockId, config)
            
            % Iterate over all block structures in the arena
            for i = 1:numel(blocks)
                
                % Extract block structure
                b = blocks(i);
                
                % -----------------------------------------------------------
                % Determine block face color based on block height.
                % This visually encodes terrain elevation.
                % -----------------------------------------------------------
                if b.h == 200
                    bc = config.COLOR_BLOCK_LOW;
                elseif b.h == 400
                    bc = config.COLOR_BLOCK_MID;
                else
                    bc = config.COLOR_BLOCK_HIGH;
                end
                
                % -----------------------------------------------------------
                % Highlight the currently occupied block by the robot.
                % Uses thicker yellow edges for emphasis.
                % -----------------------------------------------------------
                if b.id == currentBlockId
                    ec = 'y';      % Edge color
                    ew = 2;        % Edge line width
                else
                    ec = [0.3 0.3 0.3];
                    ew = 0.5;
                end
                
                % -----------------------------------------------------------
                % Render the block as a 3D box primitive.
                % Coordinates are offset so that (x,y) represents center.
                % -----------------------------------------------------------
                Renderer.drawBox( ...
                    ax, ...
                    b.x - config.BLOCK_SIZE/2, ...
                    b.y - config.BLOCK_SIZE/2, ...
                    0, ...
                    config.BLOCK_SIZE, ...
                    config.BLOCK_SIZE, ...
                    b.h, ...
                    bc, ...
                    0.7, ...
                    ec, ...
                    ew);
            end
        end
        
        
        % ===================================================================
        % Draw KFS Objects (Overview View)
        % -------------------------------------------------------------------
        % Draws all KFS cubes with color, highlighting, and textures.
        % ===================================================================
        function drawKFS(ax, arena, detectedKfs, config)
            
            % Iterate over all KFS block IDs in the arena
            for i = 1:numel(arena.kfsIds)
                
                % Retrieve block corresponding to this KFS
                kb = arena.blocks(arena.kfsIds(i));
                kfsId = arena.kfsIds(i);
                
                % KFS color assigned during arena generation
                kc = arena.kfsColors{i};
                
                % -----------------------------------------------------------
                % Highlight detected KFS with bright green edges.
                % Non-detected KFS are drawn with black edges.
                % -----------------------------------------------------------
                if detectedKfs.isKey(kfsId)
                    edgeC = [0 1 0];
                    edgeW = 3;
                else
                    edgeC = 'k';
                    edgeW = 1;
                end
                
                % -----------------------------------------------------------
                % Draw the KFS cube geometry.
                % Positioned on top of its supporting block.
                % -----------------------------------------------------------
                Renderer.drawBox( ...
                    ax, ...
                    kb.x - config.KFS_SIZE/2, ...
                    kb.y - config.KFS_SIZE/2, ...
                    kb.h, ...
                    config.KFS_SIZE, ...
                    config.KFS_SIZE, ...
                    config.KFS_SIZE, ...
                    kc, ...
                    0.5, ...
                    edgeC, ...
                    edgeW);
                
                % -----------------------------------------------------------
                % Apply texture mapping if image data exists.
                % -----------------------------------------------------------
                if i <= numel(arena.kfsImages) && ...
                   ~isempty(arena.kfsImages{i})
                    Renderer.drawTexturedKFS( ...
                        ax, kb, config.KFS_SIZE, arena.kfsImages{i});
                end
            end
        end
        
        
        % ===================================================================
        % Draw Robot Body and Heading Arrow
        % -------------------------------------------------------------------
        % Visualizes robot geometry and orientation.
        % ===================================================================
        function drawRobot(ax, robotState, config)
            
            % ---------------------------------------------------------------
            % Draw robot chassis as a 3D box centered at its position.
            % ---------------------------------------------------------------
            Renderer.drawBox( ...
                ax, ...
                robotState.position(1) - config.R2_WIDTH/2, ...
                robotState.position(2) - config.R2_LENGTH/2, ...
                robotState.position(3), ...
                config.R2_WIDTH, ...
                config.R2_LENGTH, ...
                config.R2_HEIGHT, ...
                config.COLOR_ROBOT, ...
                0.8, ...
                'k', ...
                2);
            
            % ---------------------------------------------------------------
            % Draw direction arrow representing robot yaw orientation.
            % ---------------------------------------------------------------
            arrowLen = 500;
            
            % Compute arrow endpoint using yaw angle
            arrowEnd = robotState.position + ...
                [cos(deg2rad(robotState.yaw-90)) * arrowLen, ...
                 sin(deg2rad(robotState.yaw-90)) * arrowLen, ...
                 0];
            
            % Render arrow using 3D vector plot
            quiver3( ...
                ax, ...
                robotState.position(1), ...
                robotState.position(2), ...
                robotState.position(3) + config.R2_HEIGHT/2, ...
                arrowEnd(1) - robotState.position(1), ...
                arrowEnd(2) - robotState.position(2), ...
                0, ...
                'r', 'LineWidth', 3, 'MaxHeadSize', 2);
        end
        
        
        % ===================================================================
        % Draw Detection Range Overlay
        % -------------------------------------------------------------------
        % Renders a circular ring representing detection radius.
        % ===================================================================
        function drawDetectionRange(ax, robotPos, config)
            
            % Generate angular samples for circle
            theta = linspace(0, 2*pi, 50);
            
            % Parametric circle coordinates
            detX = robotPos(1) + config.DETECTION_RANGE * cos(theta);
            detY = robotPos(2) + config.DETECTION_RANGE * sin(theta);
            
            % Plot circle at robot height plane
            plot3( ...
                ax, ...
                detX, ...
                detY, ...
                ones(size(theta)) * robotPos(3), ...
                'c--', 'LineWidth', 1);
        end
        
        
        % ===================================================================
        % Draw Blocks in Camera View
        % -------------------------------------------------------------------
        % Renders only nearby blocks for camera perspective.
        % ===================================================================
        function drawBlocksInCameraView(ax, blocks, cameraPos, config)
            
            % Iterate through all blocks
            for i = 1:numel(blocks)
                
                b = blocks(i);
                
                % Distance filtering for performance
                dist = norm(cameraPos(1:2) - [b.x, b.y]);
                
                % Skip blocks far outside camera interest range
                if dist > config.CAMERA_RANGE * 1.5
                    continue;
                end
                
                % Determine block color by height
                if b.h == 200
                    bc = config.COLOR_BLOCK_LOW;
                elseif b.h == 400
                    bc = config.COLOR_BLOCK_MID;
                else
                    bc = config.COLOR_BLOCK_HIGH;
                end
                
                % Draw block box
                Renderer.drawBox( ...
                    ax, ...
                    b.x - config.BLOCK_SIZE/2, ...
                    b.y - config.BLOCK_SIZE/2, ...
                    0, ...
                    config.BLOCK_SIZE, ...
                    config.BLOCK_SIZE, ...
                    b.h, ...
                    bc, ...
                    0.8, ...
                    [0.2 0.2 0.2], ...
                    0.5);
            end
        end
        
        
        % ===================================================================
        % Draw KFS in Camera View with Visibility Logic
        % -------------------------------------------------------------------
        % Applies occlusion checking and FOV filtering.
        % ===================================================================
        function kfsInView = drawKFSInCameraView( ...
                ax, arena, cameraPos, cameraYaw, detectedKfs, config)
            
            % Counter for number of KFS currently visible in camera FOV
            kfsInView = 0;
            
            % Convert yaw angle to radians
            yawRad = deg2rad(cameraYaw);
            
            % Compute 2D forward direction vector of camera
            cameraForward2d = [cos(yawRad - pi/2), ...
                               sin(yawRad - pi/2)];
            
            % Iterate over all KFS objects
            for i = 1:numel(arena.kfsIds)
                
                kb = arena.blocks(arena.kfsIds(i));
                kfsId = arena.kfsIds(i);
                kc = arena.kfsColors{i};
                
                % Distance check for visibility relevance
                dist = norm(cameraPos(1:2) - [kb.x, kb.y]);
                if dist > config.CAMERA_RANGE * 1.5
                    continue;
                end
                
                % Compute target point at KFS center height
                targetPt = [kb.x, kb.y, kb.h + config.KFS_SIZE/2];
                
                % Perform occlusion ray test
                isVisible = ~OcclusionChecker.isOccluded( ...
                    cameraPos, targetPt, arena.blocks, ...
                    config.BLOCK_SIZE, kb.id);
                
                % -----------------------------------------------------------
                % If visible and within detection range:
                %   - Check angular FOV alignment
                %   - Highlight accordingly
                % -----------------------------------------------------------
                if isVisible && dist <= config.DETECTION_RANGE
                    
                    % Direction vector to KFS (2D)
                    toKfs = targetPt(1:2)' - cameraPos(1:2)';
                    toKfsNorm = toKfs / norm(toKfs);
                    
                    % Angular difference between camera forward and target
                    angle = acosd(dot(cameraForward2d, toKfsNorm));
                    
                    % Within camera field-of-view
                    if angle <= config.DETECTION_FOV / 2
                        kfsInView = kfsInView + 1;
                        edgeCol = [0 1 0];
                        edgeWidth = 3;
                    else
                        edgeCol = [0.5 0.5 0.5];
                        edgeWidth = 1;
                    end
                else
                    edgeCol = [0.3 0.3 0.3];
                    edgeWidth = 0.5;
                end
                
                % Draw KFS cube
                Renderer.drawBox( ...
                    ax, ...
                    kb.x - config.KFS_SIZE/2, ...
                    kb.y - config.KFS_SIZE/2, ...
                    kb.h, ...
                    config.KFS_SIZE, ...
                    config.KFS_SIZE, ...
                    config.KFS_SIZE, ...
                    kc, ...
                    0.4, ...
                    edgeCol, ...
                    edgeWidth);
                
                % Apply texture if available
                if i <= numel(arena.kfsImages) && ...
                   ~isempty(arena.kfsImages{i})
                    Renderer.drawTexturedKFS( ...
                        ax, kb, config.KFS_SIZE, arena.kfsImages{i});
                end
            end
        end
        
        
        % ===================================================================
        % Draw Generic 3D Box Primitive
        % -------------------------------------------------------------------
        % Generates vertices and faces and renders a cuboid mesh.
        % ===================================================================
        function drawBox(ax, x, y, z, w, l, h, col, alph, edgeC, edgeW)
            
            % Compute vertices of cuboid
            V = [ ...
                x     y     z; ...
                x+w   y     z; ...
                x+w   y+l   z; ...
                x     y+l   z; ...
                x     y     z+h; ...
                x+w   y     z+h; ...
                x+w   y+l   z+h; ...
                x     y+l   z+h ];
            
            % Define faces using vertex indices
            F = [ ...
                1 2 6 5; ...
                2 3 7 6; ...
                3 4 8 7; ...
                4 1 5 8; ...
                1 2 3 4; ...
                5 6 7 8 ];
            
            % Render patch mesh
            patch( ...
                ax, ...
                'Vertices', V, ...
                'Faces', F, ...
                'FaceColor', col, ...
                'FaceAlpha', alph, ...
                'EdgeColor', edgeC, ...
                'LineWidth', edgeW);
        end
        
        
        % ===================================================================
        % Draw Textured KFS Cube Faces
        % -------------------------------------------------------------------
        % Projects image textures onto multiple faces of a cube.
        % ===================================================================
        function drawTexturedKFS(ax, kb, KFS_SIZE, img)
            
            % Size of texture projection region (slightly smaller than cube)
            texSize = 200;
            
            % Compute texture bounding coordinates
            xi0 = kb.x - texSize/2;
            xi1 = kb.x + texSize/2;
            yi0 = kb.y - texSize/2;
            yi1 = kb.y + texSize/2;
            z0  = kb.h;
            z1  = kb.h + KFS_SIZE;
            
            % ------------------ Top Face -------------------------------
            X = [xi0 xi1; xi0 xi1];
            Y = [yi0 yi0; yi1 yi1];
            Z = [z1  z1;  z1  z1];
            surf(ax, X, Y, Z, ...
                'CData', img, ...
                'FaceColor', 'texturemap', ...
                'EdgeColor', 'none', ...
                'FaceAlpha', 0.95);
            
            % ------------------ Front Face (+Y) ------------------------
            X = [xi0 xi1; xi0 xi1];
            Y = [yi1 yi1; yi1 yi1];
            Z = [z0  z0;  z1  z1];
            surf(ax, X, Y, Z, ...
                'CData', img, ...
                'FaceColor', 'texturemap', ...
                'EdgeColor', 'none', ...
                'FaceAlpha', 0.95);
            
            % ------------------ Back Face (-Y) -------------------------
            X = [xi1 xi0; xi1 xi0];
            Y = [yi0 yi0; yi0 yi0];
            Z = [z0  z0;  z1  z1];
            surf(ax, X, Y, Z, ...
                'CData', img, ...
                'FaceColor', 'texturemap', ...
                'EdgeColor', 'none', ...
                'FaceAlpha', 0.95);
            
            % ------------------ Left Face (-X) -------------------------
            X = [xi0 xi0; xi0 xi0];
            Y = [yi0 yi1; yi0 yi1];
            Z = [z0  z0;  z1  z1];
            surf(ax, X, Y, Z, ...
                'CData', img, ...
                'FaceColor', 'texturemap', ...
                'EdgeColor', 'none', ...
                'FaceAlpha', 0.95);
            
            % ------------------ Right Face (+X) ------------------------
            X = [xi1 xi1; xi1 xi1];
            Y = [yi1 yi0; yi1 yi0];
            Z = [z0  z0;  z1  z1];
            surf(ax, X, Y, Z, ...
                'CData', img, ...
                'FaceColor', 'texturemap', ...
                'EdgeColor', 'none', ...
                'FaceAlpha', 0.95);
        end
    end
end
