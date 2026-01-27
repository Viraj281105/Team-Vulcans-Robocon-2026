function robocon_arena_sim_complete
% =======================================================================
% ROBOCON 2026 – COMPLETE SIMULATION SUITE
% Features:
%   ✅ 3D Camera View
%   ✅ KFS Detection Logic (Simulated Vision Pipeline)
%   ✅ Collision Constraints
%   ✅ Robot Orientation (Yaw visualization)
%   ✅ Adjustable Camera Pitch
%   ✅ Trajectory Export (CSV/MAT)
%   ✅ Path Recording & Replay
%   ✅ Detection Statistics
% =======================================================================
% Controls:
%   W/S : Move Forward/Backward (in robot's facing direction)
%   A/D : Strafe Left/Right
%   Q/E : Rotate robot Left/Right
%   SPACE: Climb to current block height
%   C    : Reset climb (Drop to ground)
%   Z/X  : Decrease/Increase camera pitch
%   R    : Reset robot to start
%   T    : Randomize KFS locations
%   P    : Toggle path recording
%   L    : Export trajectory to file
%   H    : Toggle help overlay
%   ESC  : Quit
% =======================================================================
    
    %% ======================= INITIALIZATION =====================
    set(groot, 'ShowHiddenHandles', 'on'); 
    delete(get(groot, 'Children')); 
    clc;
    
    running = true;
    show_help = false;
    recording_path = false;
    
    % --- CONSTANTS ---
    COLOR_KFS_R1 = [1 0 0];
    COLOR_KFS_R2_REAL = [0 0 1];
    COLOR_KFS_FAKE = [1 1 0];
    COLOR_PATHWAY = [236 162 151]/255;
    COLOR_ROBOT = [0.4 0.4 0.4];
    
    KFS_SIZE = 200;
    ARENA_X = 6000;
    ARENA_Y = 7300;
    ARENA_Z = 2000;
    BLOCK_SIZE = 1200;
    FOREST_COLS = 3;
    FOREST_ROWS = 4;
    
    % Robot params
    R2_W = 800; R2_L = 800;
    R2_H_GROUND = 800;
    INITIAL_BLOCK_ID = 2;
    MOVE_SPEED = 150; % mm per step
    ROT_SPEED = 5;    % degrees per step
    
    % Camera params (adjustable)
    CAMERA_HEIGHT_OFFSET = R2_H_GROUND;
    CAMERA_FOV_H = 100;
    CAMERA_FOV_V = 65;
    camera_pitch = 15;     % Variable now!
    PITCH_STEP = 5;
    CAMERA_RANGE = 3000;
    
    % KFS Detection params (simulating vision pipeline)
    DETECTION_RANGE = 2500;
    DETECTION_FOV = 90; % degrees
    MIN_DETECTION_TIME = 0.5; % seconds to confirm detection
    CONFIDENCE_THRESHOLD = 0.50;
    
    % --- FOREST GENERATION ---
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
    
    % --- KFS CONFIGURATION ---
    all_kfs_ids = [];
    kfs_types = [];
    total_r2_kfs = 4;
    
    % --- ROBOT STATE ---
    current_block_id = INITIAL_BLOCK_ID;
    curr_b = forest_blocks(current_block_id);
    robot_pos = [curr_b.x, curr_b.y, 0];
    robot_yaw = 0; % Robot orientation
    camera_yaw = 0; % Camera can be independent or coupled
    
    % --- PATH RECORDING ---
    path_history = [];
    detection_log = struct('time',{},'kfs_id',{},'type',{},'confidence',{},'pos',{});
    mission_start_time = tic;
    
    % --- DETECTION STATE ---
    detected_kfs = containers.Map('KeyType','double','ValueType','any');
    detection_timers = containers.Map('KeyType','double','ValueType','double');
    
    % --- STATISTICS ---
    stats = struct(...
        'r2_found', 0, ...
        'r1_found', 0, ...
        'fake_found', 0, ...
        'total_distance', 0, ...
        'mission_time', 0);
    
    last_pos = robot_pos;
    
    randomizeKFS();
    
    %% ======================= FIGURE SETUP ==============================
    fig = figure('Name','ROBOCON 2026 - Complete Simulation Suite', ...
                 'Position',[50 50 1800 900], ...
                 'Color',[0.15 0.15 0.15],...
                 'KeyPressFcn',@keyHandler,...
                 'NumberTitle','off',...
                 'MenuBar','none',...
                 'ToolBar','figure');
    
    % Create 3 subplots
    ax_main = subplot(2,3,[1 4]);  % Large left panel
    ax_cam  = subplot(2,3,[2 3 5 6]); % Large right panel
    
    fprintf('=== ROBOCON 2026 - Complete Simulation ===\n');
    fprintf('All systems initialized.\n');
    fprintf('Press "H" for help, "L" to export trajectory.\n');

    %% ======================= MAIN LOOP =================================
    while running && ishandle(fig)
        
        % Update camera position (follows robot)
        camera_pos = robot_pos + [0 0 CAMERA_HEIGHT_OFFSET];
        
        % Camera yaw follows robot yaw (coupled)
        camera_yaw = robot_yaw;
        
        % Rotation matrices
        yaw_rad = deg2rad(camera_yaw);
        R_yaw = [ cos(yaw_rad) -sin(yaw_rad) 0;
                  sin(yaw_rad)  cos(yaw_rad) 0;
                  0             0            1];
        
        pitch_rad = deg2rad(camera_pitch);
        R_pitch = [ 1  0              0;
                    0  cos(pitch_rad) -sin(pitch_rad);
                    0  sin(pitch_rad)  cos(pitch_rad)];
        
        R_cam = R_yaw * R_pitch;
        
        % ==================== KFS DETECTION LOGIC ====================
        % Simulate vision pipeline detecting KFS in view
        camera_forward = R_yaw * [0; -1; 0]; % Robot looks in -Y direction
        
        for i = 1:numel(all_kfs_ids)
            kb = forest_blocks(all_kfs_ids(i));
            kt = kfs_types(i);
            kfs_id = all_kfs_ids(i);
            
            target_pt = [kb.x, kb.y, kb.h + KFS_SIZE/2];
            
            % Check if in range
            dist_to_kfs = norm(camera_pos(1:2) - target_pt(1:2));
            
            if dist_to_kfs <= DETECTION_RANGE
                % Check if in FOV
                to_kfs = target_pt(1:2)' - camera_pos(1:2)';
                to_kfs_norm = to_kfs / norm(to_kfs);
                angle = acosd(dot(camera_forward(1:2), to_kfs_norm));
                
                if angle <= DETECTION_FOV/2
                    % Check occlusion
                    is_visible = ~isOccluded(camera_pos, target_pt, forest_blocks, BLOCK_SIZE, kb.id);
                    
                    if is_visible
                        % Simulate confidence based on distance
                        confidence = 1.0 - (dist_to_kfs / DETECTION_RANGE) * 0.5;
                        confidence = max(0.5, min(1.0, confidence));
                        
                        % Start/update detection timer
                        if ~detection_timers.isKey(kfs_id)
                            detection_timers(kfs_id) = tic;
                        end
                        
                        % Confirm detection after MIN_DETECTION_TIME
                        if toc(detection_timers(kfs_id)) >= MIN_DETECTION_TIME
                            if ~detected_kfs.isKey(kfs_id)
                                detected_kfs(kfs_id) = struct('type',kt,'confidence',confidence);
                                
                                % Log detection
                                log_entry = struct(...
                                    'time', toc(mission_start_time), ...
                                    'kfs_id', kfs_id, ...
                                    'type', char(kt), ...
                                    'confidence', confidence, ...
                                    'pos', robot_pos);
                                detection_log(end+1) = log_entry;
                                
                                % Update stats
                                if kt == "R2"
                                    stats.r2_found = stats.r2_found + 1;
                                elseif kt == "R1"
                                    stats.r1_found = stats.r1_found + 1;
                                else
                                    stats.fake_found = stats.fake_found + 1;
                                end
                                
                                fprintf('[DETECT] %s KFS at Block %d (Conf: %.2f)\n', kt, kfs_id, confidence);
                            end
                        end
                    else
                        % Reset timer if occluded
                        if detection_timers.isKey(kfs_id)
                            detection_timers.remove(kfs_id);
                        end
                    end
                else
                    % Out of FOV, reset timer
                    if detection_timers.isKey(kfs_id)
                        detection_timers.remove(kfs_id);
                    end
                end
            else
                % Out of range, reset timer
                if detection_timers.isKey(kfs_id)
                    detection_timers.remove(kfs_id);
                end
            end
        end
        
        % Update mission stats
        stats.total_distance = stats.total_distance + norm(robot_pos - last_pos);
        stats.mission_time = toc(mission_start_time);
        last_pos = robot_pos;
        
        % ==================== DRAW OVERVIEW (LEFT) ====================
        set(fig, 'CurrentAxes', ax_main);
        cla(ax_main); hold on; grid on; axis equal;
        
        xlabel('X (mm)'); ylabel('Y (mm)'); zlabel('Z (mm)');
        view(45, 30);
        xlim([0 ARENA_X]); ylim([0 ARENA_Y]); zlim([0 ARENA_Z]);
        set(ax_main, 'Color', [0.05 0.05 0.1]);
        
        % Ground
        patch([0 ARENA_X ARENA_X 0],[0 0 ARENA_Y ARENA_Y],[0 0 0 0], ...
              COLOR_PATHWAY,'FaceAlpha',0.3,'EdgeColor','none');
        
        % Blocks
        for i = 1:numel(forest_blocks)
            b = forest_blocks(i);
            if b.h == 200,     bc = [41 82 16]/255;
            elseif b.h == 400, bc = [42 113 56]/255;
            else,              bc = [152 166 80]/255;
            end
            
            if b.id == current_block_id
                ec = 'y'; ew = 2;
            else
                ec = [0.3 0.3 0.3]; ew = 0.5;
            end
            
            drawBox(ax_main, b.x-BLOCK_SIZE/2, b.y-BLOCK_SIZE/2, 0, ...
                    BLOCK_SIZE, BLOCK_SIZE, b.h, bc, 0.7, ec, ew);
        end
        
        % KFS with detection status
        for i = 1:numel(all_kfs_ids)
            kb = forest_blocks(all_kfs_ids(i));
            kt = kfs_types(i);
            kfs_id = all_kfs_ids(i);
            
            if kt=="R1", kc=COLOR_KFS_R1;
            elseif kt=="R2", kc=COLOR_KFS_R2_REAL;
            else, kc=COLOR_KFS_FAKE;
            end
            
            % Highlight if detected
            if detected_kfs.isKey(kfs_id)
                edge_c = [0 1 0]; edge_w = 3;
            else
                edge_c = 'k'; edge_w = 1;
            end
            
            drawBox(ax_main, kb.x-KFS_SIZE/2, kb.y-KFS_SIZE/2, kb.h, ...
                    KFS_SIZE, KFS_SIZE, KFS_SIZE, kc, 0.9, edge_c, edge_w);
        end
        
        % Robot with orientation indicator
        drawBox(ax_main, robot_pos(1)-R2_W/2, robot_pos(2)-R2_L/2, robot_pos(3), ...
                R2_W, R2_L, R2_H_GROUND, COLOR_ROBOT, 0.8, 'k', 2);
        
        % Direction arrow
        arrow_len = 500;
        arrow_end = robot_pos + [cos(deg2rad(robot_yaw-90))*arrow_len, ...
                                  sin(deg2rad(robot_yaw-90))*arrow_len, 0];
        quiver3(robot_pos(1), robot_pos(2), robot_pos(3)+R2_H_GROUND/2, ...
                arrow_end(1)-robot_pos(1), arrow_end(2)-robot_pos(2), 0, ...
                'r', 'LineWidth', 3, 'MaxHeadSize', 2);
        
        % Detection range circle
        theta = linspace(0, 2*pi, 50);
        det_circle_x = robot_pos(1) + DETECTION_RANGE * cos(theta);
        det_circle_y = robot_pos(2) + DETECTION_RANGE * sin(theta);
        plot3(det_circle_x, det_circle_y, ones(size(theta))*robot_pos(3), ...
              'c--', 'LineWidth', 1);
        
        % Path history
        if ~isempty(path_history)
            plot3(path_history(:,1), path_history(:,2), path_history(:,3), ...
                  'g-', 'LineWidth', 2);
        end
        
        title(sprintf('Mission: %.1fs | Distance: %.1fm | Yaw: %.0f°', ...
              stats.mission_time, stats.total_distance/1000, robot_yaw), ...
              'Color', 'white');
        
        % ==================== 3D CAMERA VIEW (RIGHT) ====================
        set(fig, 'CurrentAxes', ax_cam);
        cla(ax_cam); hold on; axis equal; axis off;
        view(3);
        
        camproj(ax_cam, 'perspective');
        
        cam_target = camera_pos + R_cam * [0; -1000; 0];
        cam_up = R_cam * [0; 0; 1];
        
        campos(ax_cam, camera_pos);
        camtarget(ax_cam, cam_target');
        camup(ax_cam, cam_up');
        camva(ax_cam, CAMERA_FOV_H);
        
        axis_range = CAMERA_RANGE * 1.2;
        xlim([camera_pos(1)-axis_range, camera_pos(1)+axis_range]);
        ylim([camera_pos(2)-axis_range, camera_pos(2)+axis_range]);
        zlim([0, ARENA_Z]);
        
        set(ax_cam, 'Color', [0.5 0.6 0.8]);
        
        % Ground
        patch([0 ARENA_X ARENA_X 0],[0 0 ARENA_Y ARENA_Y],[0 0 0 0], ...
              COLOR_PATHWAY,'FaceAlpha',0.5,'EdgeColor','none');
        
        % Blocks in view
        kfs_in_view = 0;
        for i = 1:numel(forest_blocks)
            b = forest_blocks(i);
            dist = norm(camera_pos(1:2) - [b.x, b.y]);
            if dist > CAMERA_RANGE * 1.5, continue; end
            
            if b.h == 200,     bc = [41 82 16]/255;
            elseif b.h == 400, bc = [42 113 56]/255;
            else,              bc = [152 166 80]/255;
            end
            
            drawBox(ax_cam, b.x-BLOCK_SIZE/2, b.y-BLOCK_SIZE/2, 0, ...
                    BLOCK_SIZE, BLOCK_SIZE, b.h, bc, 0.8, [0.2 0.2 0.2], 0.5);
        end
        
        % KFS in view
        for i = 1:numel(all_kfs_ids)
            kb = forest_blocks(all_kfs_ids(i));
            kt = kfs_types(i);
            kfs_id = all_kfs_ids(i);
            
            dist = norm(camera_pos(1:2) - [kb.x, kb.y]);
            if dist > CAMERA_RANGE * 1.5, continue; end
            
            if kt=="R1", kc=COLOR_KFS_R1;
            elseif kt=="R2", kc=COLOR_KFS_R2_REAL;
            else, kc=COLOR_KFS_FAKE;
            end
            
            target_pt = [kb.x, kb.y, kb.h + KFS_SIZE/2];
            is_visible = ~isOccluded(camera_pos, target_pt, forest_blocks, BLOCK_SIZE, kb.id);
            
            if is_visible && dist <= DETECTION_RANGE
                % Check FOV
                to_kfs = target_pt(1:2)' - camera_pos(1:2)';
                to_kfs_norm = to_kfs / norm(to_kfs);
                camera_forward = R_yaw * [0; -1; 0];
                angle = acosd(dot(camera_forward(1:2), to_kfs_norm));
                
                if angle <= DETECTION_FOV/2
                    kfs_in_view = kfs_in_view + 1;
                    edge_col = [0 1 0];
                    edge_width = 3;
                else
                    edge_col = [0.5 0.5 0.5];
                    edge_width = 1;
                end
            else
                edge_col = [0.3 0.3 0.3];
                edge_width = 0.5;
            end
            
            drawBox(ax_cam, kb.x-KFS_SIZE/2, kb.y-KFS_SIZE/2, kb.h, ...
                    KFS_SIZE, KFS_SIZE, KFS_SIZE, kc, 0.85, edge_col, edge_width);
        end
        
        % HUD overlay
        text(0.02, 0.98, sprintf('R2 FOUND: %d/%d | R1: %d | FAKE: %d', ...
             stats.r2_found, total_r2_kfs, stats.r1_found, stats.fake_found), ...
             'Units','normalized', 'Color','lime', 'FontSize',12, 'FontWeight','bold', ...
             'BackgroundColor',[0 0 0 0.7], 'VerticalAlignment','top');
        
        text(0.02, 0.90, sprintf('YAW: %.0f°  PITCH: %.0f°  |  KFS in FOV: %d', ...
             camera_yaw, camera_pitch, kfs_in_view), ...
             'Units','normalized', 'Color','white', 'FontSize',10, ...
             'BackgroundColor',[0 0 0 0.7], 'VerticalAlignment','top');
        
        if recording_path
            text(0.98, 0.98, '● REC', 'Units','normalized', 'Color','red', ...
                 'FontSize',14, 'FontWeight','bold', 'HorizontalAlignment','right', ...
                 'BackgroundColor',[0 0 0 0.7], 'VerticalAlignment','top');
        end
        
        % Crosshair
        text(0.5, 0.5, '+', 'Units','normalized', 'Color','lime', ...
             'FontSize',24, 'HorizontalAlignment','center', 'VerticalAlignment','middle');
        
        title('CAMERA FEED', 'Color', 'white', 'FontSize', 14);
        
        % Help overlay
        if show_help
            axes(ax_main);
            help_text = sprintf([...
                ' CONTROLS:\n' ...
                ' W/S  : Move Fwd/Back\n' ...
                ' A/D  : Strafe L/R\n' ...
                ' Q/E  : Rotate L/R\n' ...
                ' SPC  : Climb\n' ...
                ' C    : Descend\n' ...
                ' Z/X  : Pitch -/+\n' ...
                ' R    : Reset\n' ...
                ' T    : Randomize\n' ...
                ' P    : Rec Path\n' ...
                ' L    : Export\n' ...
                ' H    : Help']);
            text(0.02, 0.98, help_text, 'Units','normalized', ...
                 'VerticalAlignment','top', 'Color','yellow', ...
                 'BackgroundColor',[0 0 0 0.8], 'FontName','FixedWidth', 'FontSize', 9);
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
        
        % Reset detection state
        detected_kfs = containers.Map('KeyType','double','ValueType','any');
        detection_timers = containers.Map('KeyType','double','ValueType','double');
        stats.r2_found = 0;
        stats.r1_found = 0;
        stats.fake_found = 0;
        detection_log = struct('time',{},'kfs_id',{},'type',{},'confidence',{},'pos',{});
        
        fprintf('[RANDOMIZER] New layout | R2 at: %s\n', num2str(r2_idx));
    end

    function keyHandler(~, event)
        switch event.Key
            case 'escape'
                running = false;
                
            case 'h'
                show_help = ~show_help;
                
            case 'p'
                recording_path = ~recording_path;
                if recording_path
                    fprintf('[PATH] Recording started\n');
                else
                    fprintf('[PATH] Recording stopped\n');
                end

            case 't'
                randomizeKFS();
                
            case 'l'
                exportTrajectory();
                
            case 'r'
                current_block_id = INITIAL_BLOCK_ID;
                nb = forest_blocks(INITIAL_BLOCK_ID);
                robot_pos = [nb.x, nb.y, 0];
                robot_yaw = 0;
                camera_yaw = 0;
                path_history = [];
                stats.total_distance = 0;
                mission_start_time = tic;
                fprintf('[RESET] Robot reset to start\n');
                
            case 'w' % Forward
                [new_pos, valid] = moveRobot(robot_pos, robot_yaw, MOVE_SPEED, 0);
                if valid
                    robot_pos = new_pos;
                    updatePathHistory();
                end
                
            case 's' % Backward
                [new_pos, valid] = moveRobot(robot_pos, robot_yaw, -MOVE_SPEED, 0);
                if valid
                    robot_pos = new_pos;
                    updatePathHistory();
                end
                
            case 'a' % Strafe left
                [new_pos, valid] = moveRobot(robot_pos, robot_yaw, 0, MOVE_SPEED);
                if valid
                    robot_pos = new_pos;
                    updatePathHistory();
                end
                
            case 'd' % Strafe right
                [new_pos, valid] = moveRobot(robot_pos, robot_yaw, 0, -MOVE_SPEED);
                if valid
                    robot_pos = new_pos;
                    updatePathHistory();
                end
                
            case 'space'
                % Find block at current XY position
                for k = 1:numel(forest_blocks)
                    b = forest_blocks(k);
                    if abs(robot_pos(1) - b.x) < BLOCK_SIZE/2 && ...
                       abs(robot_pos(2) - b.y) < BLOCK_SIZE/2
                        robot_pos(3) = b.h;
                        current_block_id = b.id;
                        fprintf('[CLIMB] Climbed to Block %d (H=%d)\n', b.id, b.h);
                        break;
                    end
                end
                
            case 'c'
                robot_pos(3) = 0;
                fprintf('[DESCEND] Dropped to ground\n');
                
            case 'q'
                robot_yaw = robot_yaw + ROT_SPEED;
                
            case 'e'
                robot_yaw = robot_yaw - ROT_SPEED;
                
            case 'z'
                camera_pitch = max(-45, camera_pitch - PITCH_STEP);
                fprintf('[CAMERA] Pitch: %.0f°\n', camera_pitch);
                
            case 'x'
                camera_pitch = min(45, camera_pitch + PITCH_STEP);
                fprintf('[CAMERA] Pitch: %.0f°\n', camera_pitch);
        end
    end

    function [new_pos, valid] = moveRobot(pos, yaw, fwd, side)
        % Calculate movement in robot's local frame
        yaw_rad = deg2rad(yaw - 90); % -90 because robot faces -Y
        
        dx = fwd * cos(yaw_rad) - side * sin(yaw_rad);
        dy = fwd * sin(yaw_rad) + side * cos(yaw_rad);
        
        new_pos = pos + [dx, dy, 0];
        
        % Collision check: stay within arena
        if new_pos(1) < R2_W/2 || new_pos(1) > ARENA_X - R2_W/2 || ...
           new_pos(2) < R2_L/2 || new_pos(2) > ARENA_Y - R2_L/2
            valid = false;
            fprintf('[COLLISION] Arena boundary hit\n');
            return;
        end
        
        % Check if on a valid block (simple grid check)
        on_valid_block = false;
        for k = 1:numel(forest_blocks)
            b = forest_blocks(k);
            if abs(new_pos(1) - b.x) < BLOCK_SIZE/2 && ...
               abs(new_pos(2) - b.y) < BLOCK_SIZE/2
                on_valid_block = true;
                current_block_id = b.id;
                
                % Auto-adjust Z if climbing
                if pos(3) > 10
                    new_pos(3) = b.h;
                end
                break;
            end
        end
        
        if ~on_valid_block
            valid = false;
            fprintf('[COLLISION] Cannot move off blocks\n');
            return;
        end
        
        valid = true;
    end

    function updatePathHistory()
        if recording_path
            path_history(end+1, :) = robot_pos;
        end
    end

    function exportTrajectory()
        timestamp = datestr(now, 'yyyymmdd_HHMMSS');
        
        % Export path as CSV
        if ~isempty(path_history)
            csv_file = sprintf('trajectory_%s.csv', timestamp);
            csvwrite(csv_file, path_history);
            fprintf('[EXPORT] Path saved to: %s\n', csv_file);
        end
        
        % Export detections and stats as MAT
        mat_file = sprintf('mission_data_%s.mat', timestamp);
        save(mat_file, 'path_history', 'detection_log', 'stats', ...
             'all_kfs_ids', 'kfs_types');
        fprintf('[EXPORT] Mission data saved to: %s\n', mat_file);
        
        % Print summary
        fprintf('\n=== MISSION SUMMARY ===\n');
        fprintf('Duration: %.1f seconds\n', stats.mission_time);
        fprintf('Distance: %.2f meters\n', stats.total_distance/1000);
        fprintf('R2 Found: %d/%d\n', stats.r2_found, total_r2_kfs);
        fprintf('R1 Found: %d/3\n', stats.r1_found);
        fprintf('Fake Found: %d/1\n', stats.fake_found);
        fprintf('Total Detections: %d\n', numel(detection_log));
        fprintf('======================\n\n');
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