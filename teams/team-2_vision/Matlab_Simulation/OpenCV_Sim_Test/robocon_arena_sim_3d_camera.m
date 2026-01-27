function robocon_arena_sim_3d_camera
% =======================================================================
% ROBOCON 2026 – KUNG FU QUEST
% R2 MEIHUA FOREST SIMULATOR (3D CAMERA FEED)
% =======================================================================
% Controls (Click figure to focus first):
%   W/S : Move +Y / -Y (North/South)
%   A/D : Move -X / +X (West/East)
%   SPACE: Climb to current block height
%   C    : Reset climb (Drop to ground)
%   Q/E  : Rotate camera yaw (Left/Right)
%   R    : Reset robot to start
%   T    : RANDOMIZE KFS LOCATIONS
%   H    : Toggle help overlay
%   ESC  : Quit
% =======================================================================
    %% ======================= ROBUST INITIALIZATION =====================
    set(groot, 'ShowHiddenHandles', 'on'); 
    delete(get(groot, 'Children')); 
    clc;
    
    running = true;
    show_help = false;
    
    % --- CONSTANTS & PARAMETERS ---
    COLOR_KFS_R1 = [1 0 0];
    COLOR_KFS_R2_REAL = [0 0 1];
    COLOR_KFS_FAKE = [1 1 0];
    COLOR_CAMERA_FOV = [0 1 0];
    COLOR_PATHWAY = [236 162 151]/255;
    COLOR_ROBOT = [0.4 0.4 0.4];
    
    KFS_SIZE = 200;
    ARENA_X = 6000;
    ARENA_Y = 7300;
    ARENA_Z = 2000;
    BLOCK_SIZE = 1200;
    FOREST_COLS = 3;
    FOREST_ROWS = 4;
    
    STEP_YAW = 5;
    
    R2_W = 800; R2_L = 800;
    R2_H_GROUND = 800;
    INITIAL_BLOCK_ID = 2;
    
    CAMERA_HEIGHT_OFFSET = R2_H_GROUND;
    CAMERA_FOV_H = 100;
    CAMERA_FOV_V = 65;
    CAMERA_PITCH = 15;
    CAMERA_RANGE = 3000;
    
    % --- GENERATE FOREST ---
    forest_width  = FOREST_COLS * BLOCK_SIZE;
    forest_height = FOREST_ROWS * BLOCK_SIZE;
    forest_x_min = (ARENA_X - forest_width)/2;
    forest_y_min = (ARENA_Y - forest_height)/2;
    
    h_map = [ ...
        200 200 400; ...
        200 400 600; ...
        400 600 400; ...
        200 400 200]; 
    
    forest_blocks = struct('id',{},'row',{},'col',{},'x',{},'y',{},'h',{});
    id_counter = 0;
    for r = 1:FOREST_ROWS
        for c = 1:FOREST_COLS
            id_counter = id_counter + 1;
            cx = forest_x_min + (c-0.5)*BLOCK_SIZE;
            cy = forest_y_min + (r-0.5)*BLOCK_SIZE;
            forest_blocks(id_counter).id  = id_counter;
            forest_blocks(id_counter).row = r;
            forest_blocks(id_counter).col = c;
            forest_blocks(id_counter).x   = cx;
            forest_blocks(id_counter).y   = cy;
            forest_blocks(id_counter).h   = h_map(r,c);
        end
    end
    
    all_kfs_ids = [];
    kfs_types = [];
    total_r2_kfs = 4;
    
    current_block_id = INITIAL_BLOCK_ID;
    curr_b = forest_blocks(current_block_id);
    robot_pos = [curr_b.x, curr_b.y, 0];
    camera_yaw = 0;
    
    fov_h_rad = deg2rad(CAMERA_FOV_H);
    fov_v_rad = deg2rad(CAMERA_FOV_V);
    half_h = tan(fov_h_rad/2)*CAMERA_RANGE;
    half_v = tan(fov_v_rad/2)*CAMERA_RANGE;
    
    corners_cam = [ ...
         half_h  -CAMERA_RANGE   half_v;
         half_h  -CAMERA_RANGE  -half_v;
        -half_h  -CAMERA_RANGE  -half_v;
        -half_h  -CAMERA_RANGE   half_v];

    randomizeKFS();
    
    %% ======================= FIGURE SETUP ==============================
    fig = figure('Name','R2 Meihua Forest Simulator - 3D Camera', ...
                 'Position',[50 50 1600 800], ...
                 'Color',[0.95 0.95 0.95],...
                 'KeyPressFcn',@keyHandler,...
                 'NumberTitle','off',...
                 'MenuBar','none',...
                 'ToolBar','figure');
    
    ax_main = subplot(1,2,1);
    ax_cam  = subplot(1,2,2);
    
    fprintf('=== ROBOCON 2026 - 3D Camera View Simulator ===\n');
    fprintf('Camera now shows actual 3D perspective!\n');
    fprintf('Press "T" to randomize KFS placements.\n');

    %% ======================= MAIN LOOP =================================
    while running && ishandle(fig)
        
        camera_pos = robot_pos + [0 0 CAMERA_HEIGHT_OFFSET];
        
        yaw_rad = deg2rad(camera_yaw);
        R_yaw = [ cos(yaw_rad) -sin(yaw_rad) 0;
                  sin(yaw_rad)  cos(yaw_rad) 0;
                  0             0            1];
        
        pitch_rad = deg2rad(CAMERA_PITCH);
        R_pitch = [ 1  0              0;
                    0  cos(pitch_rad) -sin(pitch_rad);
                    0  sin(pitch_rad)  cos(pitch_rad)];
        
        R_cam = R_yaw * R_pitch;
        corners_world = (R_cam * corners_cam')' + camera_pos;
        
        % ==================== DRAW OVERVIEW (LEFT) ====================
        set(fig, 'CurrentAxes', ax_main);
        cla(ax_main); hold on; grid on; axis equal;
        
        xlabel('X'); ylabel('Y'); zlabel('Z');
        view(45, 30);
        xlim([0 ARENA_X]); ylim([0 ARENA_Y]); zlim([0 ARENA_Z]);
        
        patch([0 ARENA_X ARENA_X 0],[0 0 ARENA_Y ARENA_Y],[0 0 0 0], ...
              COLOR_PATHWAY,'FaceAlpha',0.3,'EdgeColor','none');
        
        for i = 1:numel(forest_blocks)
            b = forest_blocks(i);
            if b.h == 200,     bc = [41 82 16]/255;
            elseif b.h == 400, bc = [42 113 56]/255;
            else,              bc = [152 166 80]/255;
            end
            
            if b.id == current_block_id
                ec = 'y'; ew = 2;
            else
                ec = 'k'; ew = 0.5;
            end
            
            drawBox(ax_main, b.x-BLOCK_SIZE/2, b.y-BLOCK_SIZE/2, 0, ...
                    BLOCK_SIZE, BLOCK_SIZE, b.h, bc, 0.9, ec, ew);
            
            text(b.x, b.y, b.h+100, sprintf('B%d',b.id), ...
                 'Color','w','FontWeight','bold','HorizontalAlignment','center');
        end
        
        for i = 1:numel(all_kfs_ids)
            kb = forest_blocks(all_kfs_ids(i));
            kt = kfs_types(i);
            if kt=="R1", kc=COLOR_KFS_R1;
            elseif kt=="R2", kc=COLOR_KFS_R2_REAL;
            else, kc=COLOR_KFS_FAKE;
            end
            
            drawBox(ax_main, kb.x-KFS_SIZE/2, kb.y-KFS_SIZE/2, kb.h, ...
                    KFS_SIZE, KFS_SIZE, KFS_SIZE, kc, 1.0, 'k', 1);
        end
        
        drawBox(ax_main, robot_pos(1)-R2_W/2, robot_pos(2)-R2_L/2, robot_pos(3), ...
                R2_W, R2_L, R2_H_GROUND, COLOR_ROBOT, 0.8, 'k', 2);
        
        for i = 1:4
            next = mod(i,4)+1;
            line([camera_pos(1) corners_world(i,1)], ...
                 [camera_pos(2) corners_world(i,2)], ...
                 [camera_pos(3) corners_world(i,3)], 'Color','g','LineStyle','--');
            line([corners_world(i,1) corners_world(next,1)], ...
                 [corners_world(i,2) corners_world(next,2)], ...
                 [corners_world(i,3) corners_world(next,3)], 'Color','g');
        end
        
        title(sprintf('POS: B%d | Z: %.0f | YAW: %.0f°', current_block_id, robot_pos(3), camera_yaw));
        
        % ==================== 3D CAMERA VIEW (RIGHT) ====================
        set(fig, 'CurrentAxes', ax_cam);
        cla(ax_cam); hold on; axis equal; axis off;
        view(3);
        
        % Set camera view properties for first-person perspective
        camproj(ax_cam, 'perspective');
        
        % Camera looks in -Y direction in local frame
        cam_target = camera_pos + R_cam * [0; -1000; 0];
        cam_up = R_cam * [0; 0; 1];
        
        campos(ax_cam, camera_pos);
        camtarget(ax_cam, cam_target');
        camup(ax_cam, cam_up');
        camva(ax_cam, CAMERA_FOV_H);
        
        % Set appropriate axis limits based on visible range
        axis_range = CAMERA_RANGE * 1.2;
        xlim([camera_pos(1)-axis_range, camera_pos(1)+axis_range]);
        ylim([camera_pos(2)-axis_range, camera_pos(2)+axis_range]);
        zlim([0, ARENA_Z]);
        
        % Sky/background
        set(ax_cam, 'Color', [0.6 0.7 0.9]);
        
        % Draw ground plane
        patch([0 ARENA_X ARENA_X 0],[0 0 ARENA_Y ARENA_Y],[0 0 0 0], ...
              COLOR_PATHWAY,'FaceAlpha',0.5,'EdgeColor','none');
        
        % Draw all forest blocks
        kfs_found = 0;
        for i = 1:numel(forest_blocks)
            b = forest_blocks(i);
            
            % Check if block is roughly within camera view
            dist = norm(camera_pos(1:2) - [b.x, b.y]);
            if dist > CAMERA_RANGE * 1.5, continue; end
            
            % Block color by height
            if b.h == 200,     bc = [41 82 16]/255;
            elseif b.h == 400, bc = [42 113 56]/255;
            else,              bc = [152 166 80]/255;
            end
            
            drawBox(ax_cam, b.x-BLOCK_SIZE/2, b.y-BLOCK_SIZE/2, 0, ...
                    BLOCK_SIZE, BLOCK_SIZE, b.h, bc, 0.9, [0.2 0.2 0.2], 0.5);
        end
        
        % Draw KFS objects
        for i = 1:numel(all_kfs_ids)
            kb = forest_blocks(all_kfs_ids(i));
            kt = kfs_types(i);
            
            dist = norm(camera_pos(1:2) - [kb.x, kb.y]);
            if dist > CAMERA_RANGE * 1.5, continue; end
            
            if kt=="R1", kc=COLOR_KFS_R1;
            elseif kt=="R2", kc=COLOR_KFS_R2_REAL;
            else, kc=COLOR_KFS_FAKE;
            end
            
            % Check occlusion
            target_pt = [kb.x, kb.y, kb.h + KFS_SIZE/2];
            is_visible = ~isOccluded(camera_pos, target_pt, forest_blocks, BLOCK_SIZE, kb.id);
            
            % Draw the KFS box
            if is_visible
                edge_col = 'k';
                edge_width = 2;
            else
                edge_col = [0.3 0.3 0.3];
                edge_width = 0.5;
            end
            
            drawBox(ax_cam, kb.x-KFS_SIZE/2, kb.y-KFS_SIZE/2, kb.h, ...
                    KFS_SIZE, KFS_SIZE, KFS_SIZE, kc, 0.95, edge_col, edge_width);
            
            % Count detected R2 targets
            if is_visible && kt == "R2"
                kfs_found = kfs_found + 1;
            end
        end
        
        % Add HUD overlay
        text(0.02, 0.98, sprintf('R2 DETECTED: %d / %d', kfs_found, total_r2_kfs), ...
             'Units','normalized', 'Color','lime', 'FontSize',14, 'FontWeight','bold', ...
             'BackgroundColor',[0 0 0 0.6], 'VerticalAlignment','top');
        
        text(0.02, 0.92, sprintf('YAW: %.0f°  PITCH: %.0f°', camera_yaw, CAMERA_PITCH), ...
             'Units','normalized', 'Color','white', 'FontSize',10, ...
             'BackgroundColor',[0 0 0 0.6], 'VerticalAlignment','top');
        
        % Crosshair
        text(0.5, 0.5, '+', 'Units','normalized', 'Color','lime', ...
             'FontSize',20, 'HorizontalAlignment','center', 'VerticalAlignment','middle');
        
        title('CAMERA VIEW (3D Perspective)', 'Color', 'white');
        
        % Help overlay
        if show_help
            axes(ax_main);
            text(0.05, 0.95, getHelpString(), 'Units','normalized', ...
                 'VerticalAlignment','top', 'Color','y', 'BackgroundColor',[0 0 0 0.7], ...
                 'FontName','FixedWidth');
        end
        
        drawnow;
    end
    
    %% ======================= NESTED FUNCTIONS ==========================
    
    function randomizeKFS()
        pool = randperm(12);
        r1_idx = pool(1:3);
        r2_idx = pool(4:7);
        fake_idx = pool(8);
        
        all_kfs_ids = [r1_idx r2_idx fake_idx];
        kfs_types = [repmat("R1",1,3), repmat("R2",1,4), "FAKE"];
        
        fprintf('[RANDOMIZER] New Layout Generated!\n');
        fprintf('  R2 Targets at Blocks: %s\n', num2str(r2_idx));
    end

    function keyHandler(~, event)
        switch event.Key
            case 'escape'
                running = false;
                
            case 'h'
                show_help = ~show_help;

            case 't'
                randomizeKFS();
                
            case 'r'
                current_block_id = INITIAL_BLOCK_ID;
                nb = forest_blocks(INITIAL_BLOCK_ID);
                robot_pos = [nb.x, nb.y, 0];
                camera_yaw = 0;
                
            case {'w','s','a','d'}
                cb = forest_blocks(current_block_id);
                r = cb.row; c = cb.col;
                
                if strcmp(event.Key, 'w'), r = min(r+1, FOREST_ROWS); end
                if strcmp(event.Key, 's'), r = max(r-1, 1); end
                if strcmp(event.Key, 'a'), c = max(c-1, 1); end
                if strcmp(event.Key, 'd'), c = min(c+1, FOREST_COLS); end
                
                for k = 1:numel(forest_blocks)
                    if forest_blocks(k).row == r && forest_blocks(k).col == c
                        current_block_id = k;
                        nb = forest_blocks(k);
                        
                        if robot_pos(3) > 10
                            new_z = nb.h;
                        else
                            new_z = 0;
                        end
                        
                        robot_pos = [nb.x, nb.y, new_z];
                        break;
                    end
                end
                
            case 'space'
                cb = forest_blocks(current_block_id);
                robot_pos(3) = cb.h;
                
            case 'c'
                robot_pos(3) = 0;
                
            case 'q'
                camera_yaw = camera_yaw + STEP_YAW;
                
            case 'e'
                camera_yaw = camera_yaw - STEP_YAW;
        end
    end
end

%% ======================= LOCAL HELPER FUNCTIONS =======================
function drawBox(ax, x, y, z, w, l, h, col, alph, edge_c, edge_w)
    V = [x y z; x+w y z; x+w y+l z; x y+l z;
         x y z+h; x+w y z+h; x+w y+l z+h; x y+l z+h];
    F = [1 2 6 5; 2 3 7 6; 3 4 8 7; 4 1 5 8; 1 2 3 4; 5 6 7 8];
    patch(ax, 'Vertices',V,'Faces',F,'FaceColor',col,'FaceAlpha',alph,...
          'EdgeColor',edge_c,'LineWidth',edge_w);
end

function occ = isOccluded(p1, p2, blocks, b_size, target_id)
    occ = false;
    for k = 1:numel(blocks)
        if blocks(k).id == target_id, continue; end
        
        b = blocks(k);
        min_b = [b.x - b_size/2, b.y - b_size/2, 0];
        max_b = [b.x + b_size/2, b.y + b_size/2, b.h + 50]; 
        
        if rayBoxIntersect(p1, p2, min_b, max_b)
            occ = true; return;
        end
    end
end

function hit = rayBoxIntersect(p1, p2, min_b, max_b)
    dir = p2 - p1;
    tmin = 0; tmax = 1;
    
    for i = 1:3
        if abs(dir(i)) < 1e-6
            if p1(i) < min_b(i) || p1(i) > max_b(i)
                hit = false; return;
            end
        else
            invD = 1.0 / dir(i);
            t1 = (min_b(i) - p1(i)) * invD;
            t2 = (max_b(i) - p1(i)) * invD;
            if t1 > t2, tmp=t1; t1=t2; t2=tmp; end
            tmin = max(tmin, t1);
            tmax = min(tmax, t2);
            if tmin > tmax, hit = false; return; end
        end
    end
    hit = true;
end

function txt = getHelpString()
    txt = sprintf(' CONTROLS:\n W/S: Move Y\n A/D: Move X\n SPC: Climb\n C  : Down\n Q/E: Rotate\n R  : Reset\n T  : Randomize');
end