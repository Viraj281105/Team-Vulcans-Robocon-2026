classdef Renderer
    % RENDERER Utility functions for rendering 3D scene elements
    
    methods (Static)
        function drawBlocks(ax, blocks, currentBlockId, config)
            % Draw all forest blocks
            for i = 1:numel(blocks)
                b = blocks(i);
                
                % Color based on height
                if b.h == 200
                    bc = config.COLOR_BLOCK_LOW;
                elseif b.h == 400
                    bc = config.COLOR_BLOCK_MID;
                else
                    bc = config.COLOR_BLOCK_HIGH;
                end
                
                % Edge color for current block
                if b.id == currentBlockId
                    ec = 'y';
                    ew = 2;
                else
                    ec = [0.3 0.3 0.3];
                    ew = 0.5;
                end
                
                Renderer.drawBox(ax, b.x - config.BLOCK_SIZE/2, ...
                    b.y - config.BLOCK_SIZE/2, 0, ...
                    config.BLOCK_SIZE, config.BLOCK_SIZE, b.h, ...
                    bc, 0.7, ec, ew);
            end
        end
        
        function drawKFS(ax, arena, detectedKfs, config)
            % Draw KFS cubes with textures
            for i = 1:numel(arena.kfsIds)
                kb = arena.blocks(arena.kfsIds(i));
                kfsId = arena.kfsIds(i);
                kc = arena.kfsColors{i};
                
                % Highlight if detected
                if detectedKfs.isKey(kfsId)
                    edgeC = [0 1 0];
                    edgeW = 3;
                else
                    edgeC = 'k';
                    edgeW = 1;
                end
                
                Renderer.drawBox(ax, kb.x - config.KFS_SIZE/2, ...
                    kb.y - config.KFS_SIZE/2, kb.h, ...
                    config.KFS_SIZE, config.KFS_SIZE, config.KFS_SIZE, ...
                    kc, 0.5, edgeC, edgeW);
                
                % Draw texture
                if i <= numel(arena.kfsImages) && ~isempty(arena.kfsImages{i})
                    Renderer.drawTexturedKFS(ax, kb, config.KFS_SIZE, arena.kfsImages{i});
                end
            end
        end
        
        function drawRobot(ax, robotState, config)
            % Draw robot box with direction arrow
            Renderer.drawBox(ax, robotState.position(1) - config.R2_WIDTH/2, ...
                robotState.position(2) - config.R2_LENGTH/2, ...
                robotState.position(3), ...
                config.R2_WIDTH, config.R2_LENGTH, config.R2_HEIGHT, ...
                config.COLOR_ROBOT, 0.8, 'k', 2);
            
            % Direction arrow
            arrowLen = 500;
            arrowEnd = robotState.position + ...
                [cos(deg2rad(robotState.yaw-90))*arrowLen, ...
                 sin(deg2rad(robotState.yaw-90))*arrowLen, 0];
            quiver3(ax, robotState.position(1), robotState.position(2), ...
                robotState.position(3) + config.R2_HEIGHT/2, ...
                arrowEnd(1) - robotState.position(1), ...
                arrowEnd(2) - robotState.position(2), 0, ...
                'r', 'LineWidth', 3, 'MaxHeadSize', 2);
        end
        
        function drawDetectionRange(ax, robotPos, config)
            % Draw detection range circle
            theta = linspace(0, 2*pi, 50);
            detX = robotPos(1) + config.DETECTION_RANGE * cos(theta);
            detY = robotPos(2) + config.DETECTION_RANGE * sin(theta);
            plot3(ax, detX, detY, ones(size(theta)) * robotPos(3), ...
                'c--', 'LineWidth', 1);
        end
        
        function drawBlocksInCameraView(ax, blocks, cameraPos, config)
            % Draw blocks visible in camera view
            for i = 1:numel(blocks)
                b = blocks(i);
                dist = norm(cameraPos(1:2) - [b.x, b.y]);
                
                if dist > config.CAMERA_RANGE * 1.5
                    continue;
                end
                
                if b.h == 200
                    bc = config.COLOR_BLOCK_LOW;
                elseif b.h == 400
                    bc = config.COLOR_BLOCK_MID;
                else
                    bc = config.COLOR_BLOCK_HIGH;
                end
                
                Renderer.drawBox(ax, b.x - config.BLOCK_SIZE/2, ...
                    b.y - config.BLOCK_SIZE/2, 0, ...
                    config.BLOCK_SIZE, config.BLOCK_SIZE, b.h, ...
                    bc, 0.8, [0.2 0.2 0.2], 0.5);
            end
        end
        
        function kfsInView = drawKFSInCameraView(ax, arena, cameraPos, ...
                cameraYaw, detectedKfs, config)
            % Draw KFS in camera view with visibility highlighting
            kfsInView = 0;
            
            yawRad = deg2rad(cameraYaw);
            cameraForward2d = [cos(yawRad - pi/2), sin(yawRad - pi/2)];
            
            for i = 1:numel(arena.kfsIds)
                kb = arena.blocks(arena.kfsIds(i));
                kfsId = arena.kfsIds(i);
                kc = arena.kfsColors{i};
                
                dist = norm(cameraPos(1:2) - [kb.x, kb.y]);
                if dist > config.CAMERA_RANGE * 1.5
                    continue;
                end
                
                targetPt = [kb.x, kb.y, kb.h + config.KFS_SIZE/2];
                isVisible = ~OcclusionChecker.isOccluded(cameraPos, targetPt, ...
                    arena.blocks, config.BLOCK_SIZE, kb.id);
                
                if isVisible && dist <= config.DETECTION_RANGE
                    toKfs = targetPt(1:2)' - cameraPos(1:2)';
                    toKfsNorm = toKfs / norm(toKfs);
                    angle = acosd(dot(cameraForward2d, toKfsNorm));
                    
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
                
                Renderer.drawBox(ax, kb.x - config.KFS_SIZE/2, ...
                    kb.y - config.KFS_SIZE/2, kb.h, ...
                    config.KFS_SIZE, config.KFS_SIZE, config.KFS_SIZE, ...
                    kc, 0.4, edgeCol, edgeWidth);
                
                if i <= numel(arena.kfsImages) && ~isempty(arena.kfsImages{i})
                    Renderer.drawTexturedKFS(ax, kb, config.KFS_SIZE, arena.kfsImages{i});
                end
            end
        end
        
        function drawBox(ax, x, y, z, w, l, h, col, alph, edgeC, edgeW)
            % Draw a 3D box
            V = [x   y   z;   x+w y   z;   x+w y+l z;   x   y+l z; ...
                 x   y   z+h; x+w y   z+h; x+w y+l z+h; x   y+l z+h];
            F = [1 2 6 5; 2 3 7 6; 3 4 8 7; 4 1 5 8; 1 2 3 4; 5 6 7 8];
            patch(ax, 'Vertices', V, 'Faces', F, 'FaceColor', col, ...
                'FaceAlpha', alph, 'EdgeColor', edgeC, 'LineWidth', edgeW);
        end
        
        function drawTexturedKFS(ax, kb, KFS_SIZE, img)
            % Draw textured faces on KFS cube
            texSize = 200;
            xi0 = kb.x - texSize/2;
            xi1 = kb.x + texSize/2;
            yi0 = kb.y - texSize/2;
            yi1 = kb.y + texSize/2;
            z0 = kb.h;
            z1 = kb.h + KFS_SIZE;
            
            % Top face
            X = [xi0 xi1; xi0 xi1];
            Y = [yi0 yi0; yi1 yi1];
            Z = [z1  z1;  z1  z1];
            surf(ax, X, Y, Z, 'CData', img, 'FaceColor', 'texturemap', ...
                'EdgeColor', 'none', 'FaceAlpha', 0.95);
            
            % Front (+Y)
            X = [xi0 xi1; xi0 xi1];
            Y = [yi1 yi1; yi1 yi1];
            Z = [z0  z0;  z1  z1];
            surf(ax, X, Y, Z, 'CData', img, 'FaceColor', 'texturemap', ...
                'EdgeColor', 'none', 'FaceAlpha', 0.95);
            
            % Back (-Y)
            X = [xi1 xi0; xi1 xi0];
            Y = [yi0 yi0; yi0 yi0];
            Z = [z0  z0;  z1  z1];
            surf(ax, X, Y, Z, 'CData', img, 'FaceColor', 'texturemap', ...
                'EdgeColor', 'none', 'FaceAlpha', 0.95);
            
            % Left (-X)
            X = [xi0 xi0; xi0 xi0];
            Y = [yi0 yi1; yi0 yi1];
            Z = [z0  z0;  z1  z1];
            surf(ax, X, Y, Z, 'CData', img, 'FaceColor', 'texturemap', ...
                'EdgeColor', 'none', 'FaceAlpha', 0.95);
            
            % Right (+X)
            X = [xi1 xi1; xi1 xi1];
            Y = [yi1 yi0; yi1 yi0];
            Z = [z0  z0;  z1  z1];
            surf(ax, X, Y, Z, 'CData', img, 'FaceColor', 'texturemap', ...
                'EdgeColor', 'none', 'FaceAlpha', 0.95);
        end
    end
end
