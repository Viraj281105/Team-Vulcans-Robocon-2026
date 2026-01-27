function robocon_arena_sim
% =======================================================================
% ROBOCON 2026 – KUNG FU QUEST
% R2 MEIHUA FOREST SIMULATOR (FINAL: TILTED CAMERA + WIDE FOV)
% =======================================================================
% Controls (Click figure to focus first):
%   W/S : Move +Y / -Y (North/South)
%   A/D : Move -X / +X (West/East)
%   SPACE: Climb to current block height
%   C    : Reset climb (Drop to ground)
%   Q/E  : Rotate camera yaw (Left/Right)
%   R    : Reset robot to start
%   H    : Toggle help overlay
%   ESC  : Quit
% =======================================================================

    %% ======================= ROBUST INITIALIZATION =====================
    % 1. Force close any lingering figures avoiding callback errors
    set(groot, 'ShowHiddenHandles', 'on'); 
    delete(get(groot, 'Children')); 
    clc;
    
    % --- SHARED STATE VARIABLES (Accessible by loop and keys) ---
    running = true;
    show_help = false;
    
    % --- CONSTANTS & PARAMETERS ---
    % Colors
    COLOR_KFS_R1 = [1 0 0];            % Red
    COLOR_KFS_R2_REAL = [0 0 1];       % Blue
    COLOR_KFS_FAKE = [1 1 0];          % Yellow
    COLOR_CAMERA_FOV = [0 1 0];        % Green
    COLOR_PATHWAY = [236 162 151]/255; % Pinkish
    COLOR_ROBOT = [0.4 0.4 0.4];       % Gray
    
    % Dimensions
    KFS_SIZE = 200;
    ARENA_X = 6000;
    ARENA_Y = 7300;
    ARENA_Z = 2000;
    BLOCK_SIZE = 1200;
    FOREST_COLS = 3;
    FOREST_ROWS = 4;
    
    STEP_YAW = 5; % Degrees per press
    
    % Robot Params
    R2_W = 800; R2_L = 800;
    R2_H_GROUND = 800;
    INITIAL_BLOCK_ID = 2;
    
    % Camera Params (UPDATED)
    CAMERA_HEIGHT_OFFSET = R2_H_GROUND;
    CAMERA_FOV_H = 100;     % Horizontal FOV
    CAMERA_FOV_V = 65;      % Vertical FOV
    CAMERA_PITCH = 15;      % 30 Degrees Downward Tilt
    CAMERA_RANGE = 3000;

    % --- GENERATE FOREST ---
    forest_width  = FOREST_COLS * BLOCK_SIZE;
    forest_height = FOREST_ROWS * BLOCK_SIZE;
    forest_x_min = (ARENA_X - forest_width)/2;
    forest_y_min = (ARENA_Y - forest_height)/2;

    % Height Map (Entrance at Row 1)
    h_map = [ ...
        200 200 400; ...
        200 400 600; ...
        400 600 400; ...
        200 400 200]; 

    % Build Blocks Struct
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

    % --- KFS CONFIGURATION ---
    R1_ids = [1 3 11];
    R2_ids = [5 6 7 9];
    Fake_id = 8;
    all_kfs_ids = [R1_ids R2_ids Fake_id];
    kfs_types = [repmat("R1",1,numel(R1_ids)), ...
                 repmat("R2",1,numel(R2_ids)), ...
                 "FAKE"];
    total_r2_kfs = numel(R2_ids);

    % --- ROBOT STATE INITIALIZATION ---
    current_block_id = INITIAL_BLOCK_ID;
    curr_b = forest_blocks(current_block_id);
    robot_pos = [curr_b.x, curr_b.y, 0]; % X, Y, Z
    camera_yaw = 0;
    
    % --- CAMERA GEOMETRY PRE-CALCULATION ---
    fov_h_rad = deg2rad(CAMERA_FOV_H);
    fov_v_rad = deg2rad(CAMERA_FOV_V);
    half_h = tan(fov_h_rad/2)*CAMERA_RANGE;
    half_v = tan(fov_v_rad/2)*CAMERA_RANGE;
    % Base Frustum (Forward is -Y in Camera Local Frame)
    corners_cam = [ ...
         half_h  -CAMERA_RANGE   half_v;
         half_h  -CAMERA_RANGE  -half_v;
        -half_h  -CAMERA_RANGE  -half_v;
        -half_h  -CAMERA_RANGE   half_v];
    
    % Image Plane Setup
    img_w = 640; img_h = 360; 
    focal = img_w / (2*tan(fov_h_rad/2));
    base_img = ones(img_h,img_w,3)*0.85;

    %% ======================= FIGURE SETUP ==============================
    fig = figure('Name','R2 Meihua Forest Simulator - ROBOCON 2026', ...
                 'Position',[50 50 1400 800], ...
                 'Color',[0.95 0.95 0.95],...
                 'KeyPressFcn',@keyHandler,...
                 'NumberTitle','off',...
                 'MenuBar','none',...
                 'ToolBar','figure');
    
    ax_main = subplot(1,2,1);
    ax_cam  = subplot(1,2,2);

    fprintf('=== ROBOCON 2026 - Meihua Forest Simulator ===\n');
    fprintf('Controls active. Click figure to focus.\n');

    %% ======================= MAIN LOOP =================================
    while running && ishandle(fig)
        
        % 1. Update Camera Pose
        camera_pos = robot_pos + [0 0 CAMERA_HEIGHT_OFFSET];
        
        % --- CALC ROTATION (Yaw + Pitch) ---
        % Yaw (Z-axis rotation)
        yaw_rad = deg2rad(camera_yaw);
        R_yaw = [ cos(yaw_rad) -sin(yaw_rad) 0;
                  sin(yaw_rad)  cos(yaw_rad) 0;
                  0             0            1];

        % Pitch (X-axis rotation) - Positive tilts camera down
        pitch_rad = deg2rad(CAMERA_PITCH);
        R_pitch = [ 1  0              0;
                    0  cos(pitch_rad) -sin(pitch_rad);
                    0  sin(pitch_rad)  cos(pitch_rad)];
        
        % Combined Rotation
        R_cam = R_yaw * R_pitch; 
        
        % Transform Frustum to World using Combined Rotation
        corners_world = (R_cam * corners_cam')' + camera_pos;

        % 2. DRAW 3D SCENE
        set(fig, 'CurrentAxes', ax_main);
        cla(ax_main); hold on; grid on; axis equal;
        
        % Setup View
        xlabel('X'); ylabel('Y'); zlabel('Z');
        view(45, 30);
        xlim([0 ARENA_X]); ylim([0 ARENA_Y]); zlim([0 ARENA_Z]);
        
        % Draw Pathway
        patch([0 ARENA_X ARENA_X 0],[0 0 ARENA_Y ARENA_Y],[0 0 0 0], ...
              COLOR_PATHWAY,'FaceAlpha',0.3,'EdgeColor','none');

        % Draw Forest Blocks
        for i = 1:numel(forest_blocks)
            b = forest_blocks(i);
            % Color by height
            if b.h == 200,     bc = [41 82 16]/255;
            elseif b.h == 400, bc = [42 113 56]/255;
            else,              bc = [152 166 80]/255;
            end
            
            % Highlight current
            if b.id == current_block_id
                ec = 'y'; ew = 2;
            else
                ec = 'k'; ew = 0.5;
            end
            
            drawBox(ax_main, b.x-BLOCK_SIZE/2, b.y-BLOCK_SIZE/2, 0, ...
                    BLOCK_SIZE, BLOCK_SIZE, b.h, bc, 0.9, ec, ew);
            
            % ID Label
            text(b.x, b.y, b.h+100, sprintf('B%d',b.id), ...
                 'Color','w','FontWeight','bold','HorizontalAlignment','center');
        end

        % Draw KFS
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

        % Draw Robot
        drawBox(ax_main, robot_pos(1)-R2_W/2, robot_pos(2)-R2_L/2, robot_pos(3), ...
                R2_W, R2_L, R2_H_GROUND, COLOR_ROBOT, 0.8, 'k', 2);
        
        % Draw Camera Frustum
        for i = 1:4
            next = mod(i,4)+1;
            line([camera_pos(1) corners_world(i,1)], ...
                 [camera_pos(2) corners_world(i,2)], ...
                 [camera_pos(3) corners_world(i,3)], 'Color','g','LineStyle','--');
            line([corners_world(i,1) corners_world(next,1)], ...
                 [corners_world(i,2) corners_world(next,2)], ...
                 [corners_world(i,3) corners_world(next,3)], 'Color','g');
        end
        
        title(sprintf('POS: B%d | Z: %.0f | YAW: %.0f', current_block_id, robot_pos(3), camera_yaw));

        % 3. SIMULATE CAMERA VIEW
        set(fig, 'CurrentAxes', ax_cam);
        cla(ax_cam); axis off;
        imshow(base_img); hold on;
        
        kfs_found = 0;
        
        for i = 1:numel(all_kfs_ids)
            kb = forest_blocks(all_kfs_ids(i));
            kt = kfs_types(i);
            
            if kt=="R1", kc=COLOR_KFS_R1;
            elseif kt=="R2", kc=COLOR_KFS_R2_REAL;
            else, kc=COLOR_KFS_FAKE;
            end
            
            % Target Center
            target_pt = [kb.x, kb.y, kb.h + KFS_SIZE/2];
            
            % Occlusion Check
            if ~isOccluded(camera_pos, target_pt, forest_blocks, BLOCK_SIZE, kb.id)
                % Project using combined R_cam
                [u, v, vis] = projectPoint(target_pt, camera_pos, R_cam, focal, img_w, img_h);
                
                if vis
                    % Draw detection
                    bs = 40;
                    rectangle('Position',[u-bs/2, v-bs/2, bs, bs], 'EdgeColor',kc, 'LineWidth',3);
                    text(u, v-bs, sprintf('%s',kt), 'Color',kc, 'FontWeight','bold', 'FontSize',12, 'HorizontalAlignment','center');
                    
                    if kt == "R2", kfs_found = kfs_found + 1; end
                end
            end
        end
        
        title(sprintf('CAMERA VIEW (Pitch %d°) | R2 Found: %d / %d', CAMERA_PITCH, kfs_found, total_r2_kfs));

        % Help Overlay
        if show_help
            axes(ax_main);
            text(0.05, 0.95, getHelpString(), 'Units','normalized', ...
                 'VerticalAlignment','top', 'Color','y', 'BackgroundColor',[0 0 0 0.7], ...
                 'FontName','FixedWidth');
        end

        drawnow; % Force update
    end
    
    %% ======================= NESTED FUNCTIONS ==========================
    
    function keyHandler(~, event)
        % This function now shares scope with the main variables directly
        
        switch event.Key
            case 'escape'
                running = false;
                
            case 'h'
                show_help = ~show_help;
                
            case 'r'
                current_block_id = INITIAL_BLOCK_ID;
                nb = forest_blocks(INITIAL_BLOCK_ID);
                robot_pos = [nb.x, nb.y, 0];
                camera_yaw = 0;
                
            case {'w','s','a','d'}
                % Get current grid pos
                cb = forest_blocks(current_block_id);
                r = cb.row; c = cb.col;
                
                if strcmp(event.Key, 'w'), r = min(r+1, FOREST_ROWS); end
                if strcmp(event.Key, 's'), r = max(r-1, 1); end
                if strcmp(event.Key, 'a'), c = max(c-1, 1); end
                if strcmp(event.Key, 'd'), c = min(c+1, FOREST_COLS); end
                
                % Find new block ID
                for k = 1:numel(forest_blocks)
                    if forest_blocks(k).row == r && forest_blocks(k).col == c
                        current_block_id = k;
                        nb = forest_blocks(k);
                        
                        % Logic: If we are climbing (Z > 0), snap to new block height
                        % If we are on ground (Z == 0), stay on ground
                        if robot_pos(3) > 10 % Tolerance
                            new_z = nb.h;
                        else
                            new_z = 0;
                        end
                        
                        robot_pos = [nb.x, nb.y, new_z];
                        break;
                    end
                end
                
            case 'space'
                % Climb
                cb = forest_blocks(current_block_id);
                robot_pos(3) = cb.h;
                
            case 'c'
                % Descend
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
    patch(ax, 'Vertices',V,'Faces',F,'FaceColor',col,'FaceAlpha',alph,'EdgeColor',edge_c,'LineWidth',edge_w);
end

function [u, v, visible] = projectPoint(p, cam_pos, R, f, iw, ih)
    % Transform world point to camera frame
    v_world = p - cam_pos;
    v_cam = R' * v_world'; % Transpose rotation for inverse
    
    x = v_cam(1); y = v_cam(2); z = v_cam(3);
    
    % Camera points -Y
    depth = -y;
    
    if depth <= 100 % Near clip
        u=0; v=0; visible=false; return;
    end
    
    u = f * (x / depth) + iw/2;
    v = ih/2 - f * (z / depth);
    
    visible = (u > 0 && u < iw && v > 0 && v < ih);
end

function occ = isOccluded(p1, p2, blocks, b_size, target_id)
    occ = false;
    % Ray-Box intersection for every other block
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
    txt = sprintf(' CONTROLS:\n W/S: Move Y\n A/D: Move X\n SPC: Climb\n C  : Down\n Q/E: Rotate\n R  : Reset');
end