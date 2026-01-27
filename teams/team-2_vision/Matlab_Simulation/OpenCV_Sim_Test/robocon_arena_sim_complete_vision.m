function robocon_arena_sim_complete_vision
% =======================================================================
% ROBOCON 2026 – COMPLETE SIMULATION WITH VISION PIPELINE
% Features:
%   ✅ Oracle bone character textures on ALL 5 faces of KFS (200x200mm)
%   ✅ 4 R2 real, 3 R1, 1 fake, rest blank
%   ✅ R1 only on edges/corners of the forest
%   ✅ R2 real from dataset\real, R1 from dataset\R1, fake from dataset\fake
%   ✅ Simulated HSV + ML vision pipeline (no webcam needed)
%   ✅ Auto camera tilt to center KFS
%   ✅ Independent camera pan control
%   ✅ Robot can climb blocks with SPACE and traverse higher blocks
%   ✅ Real/Fake use same color (blue/red, random per KFS)
% =======================================================================
% Controls:
%   W/S : Move Forward/Backward
%   A/D : Strafe Left/Right
%   Q/E : Rotate robot Left/Right
%   SPACE: Climb to current block
%   C    : Descend to ground
%   Z/X  : Manual camera pitch -/+
%   I/K  : Camera pan (yaw) independent of robot
%   V    : Toggle auto-tilt (camera centers on nearest KFS)
%   R    : Reset robot
%   T    : Randomize KFS
%   P    : Toggle path recording
%   L    : Export trajectory
%   H    : Toggle help
%   ESC  : Quit
% =======================================================================

%% ======================= INITIALIZATION =====================
set(groot, 'ShowHiddenHandles', 'on');
delete(get(groot, 'Children'));
clc;

running            = true;
show_help          = false;
recording_path     = false;
auto_tilt_enabled  = false;

% --- PATHS TO ORACLE BONE IMAGES ---
DATASET_PATH = 'D:\Robotics Club\Robocon2026\Team-Vulcans-Robocon-2026\teams\team-2_vision\dataset';
R1_FOLDER    = fullfile(DATASET_PATH, 'R1');    % R1 scrolls
REAL_FOLDER  = fullfile(DATASET_PATH, 'real');  % R2 real
FAKE_FOLDER  = fullfile(DATASET_PATH, 'fake');  % R2 fake

% --- KFS COUNTS ---
total_r2_real = 4;
total_r1      = 3;
total_fake    = 1;

% --- CONSTANTS ---
COLOR_KFS_BASE_RED  = [1 0 0];
COLOR_KFS_BASE_BLUE = [0 0 1];

COLOR_PATHWAY = [236 162 151]/255;
COLOR_ROBOT   = [0.4 0.4 0.4];

KFS_SIZE    = 350;  % mm
ARENA_X     = 6000;
ARENA_Y     = 7300;
ARENA_Z     = 2000;
BLOCK_SIZE  = 1200;
FOREST_COLS = 3;
FOREST_ROWS = 4;

% Robot params
R2_W = 800; R2_L = 800;
R2_H_GROUND = 800;
INITIAL_BLOCK_ID = 2;
MOVE_SPEED = 150;
ROT_SPEED  = 5;

% Camera params
CAMERA_HEIGHT_OFFSET = R2_H_GROUND;
CAMERA_FOV_H   = 100;
CAMERA_FOV_V   = 65;
camera_pitch   = 15;
camera_yaw_offset = 0;
PITCH_STEP     = 5;
YAW_STEP       = 5;
CAMERA_RANGE   = 3000;

% Vision pipeline params
CONF_THRESHOLD     = 0.50;
DETECTION_RANGE    = 2500;
DETECTION_FOV      = 90;
MIN_DETECTION_TIME = 0.5;

% --- LOAD ORACLE BONE IMAGES ---
fprintf('[INFO] Loading oracle bone character images...\n');
[r1_images, ~]      = loadOracleBoneImages(R1_FOLDER, []);      % R1
[r2_real_images, ~] = loadOracleBoneImages(REAL_FOLDER, []);    % R2 real
[~, r2_fake_images] = loadOracleBoneImages([], FAKE_FOLDER);    % R2 fake

if isempty(r1_images)
    fprintf('[WARN] No R1 images found, using placeholder\n');
    r1_images = {ones(256,256,3)*0.2};
end
if isempty(r2_real_images)
    fprintf('[WARN] No R2 REAL images found, using placeholder\n');
    r2_real_images = {ones(256,256,3)*0.4};
end
if isempty(r2_fake_images)
    fprintf('[WARN] No R2 FAKE images found, using placeholder\n');
    r2_fake_images = {ones(256,256,3)*0.8};
end

% --- FOREST GENERATION ---
forest_width  = FOREST_COLS * BLOCK_SIZE;
forest_height = FOREST_ROWS * BLOCK_SIZE;
forest_x_min  = (ARENA_X - forest_width)/2;
forest_y_min  = (ARENA_Y - forest_height)/2;

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

% --- KFS STATE ---
all_kfs_ids = [];
kfs_types   = [];
kfs_images  = {};
kfs_colors  = {};

% --- ROBOT STATE ---
current_block_id = INITIAL_BLOCK_ID;
curr_b    = forest_blocks(current_block_id);
robot_pos = [curr_b.x, curr_b.y, 0];
robot_yaw = 180;

% --- PATH & DETECTION ---
path_history       = [];
detection_log      = struct('time',{},'kfs_id',{},'type',{},'confidence',{},'pos',{});
mission_start_time = tic;

detected_kfs     = containers.Map('KeyType','double','ValueType','any');
detection_timers = containers.Map('KeyType','double','ValueType','uint64');

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
fig = figure('Name','ROBOCON 2026 - Vision Pipeline Simulation', ...
             'Position',[50 50 1800 900], ...
             'Color',[0.15 0.15 0.15], ...
             'KeyPressFcn',@keyHandler, ...
             'NumberTitle','off', ...
             'MenuBar','none', ...
             'ToolBar','figure');

ax_main = subplot(2,3,[1 4]);
ax_cam  = subplot(2,3,[2 3 5 6]);

fprintf('=== ROBOCON 2026 - Vision Pipeline Simulation ===\n');
fprintf('Oracle bone characters on ALL KFS faces (200x200mm center)\n');
fprintf('Press "V" to toggle auto-tilt, "H" for help\n');

%% ======================= MAIN LOOP =================================
while running && ishandle(fig)

    % Camera pose
    camera_pos = robot_pos + [0 0 CAMERA_HEIGHT_OFFSET];
    camera_yaw = robot_yaw + camera_yaw_offset;

    if auto_tilt_enabled
        camera_pitch = autoAdjustPitch(camera_pos, camera_yaw);
    end

    yaw_rad   = deg2rad(camera_yaw);
    pitch_rad = deg2rad(camera_pitch);

    R_yaw = [ cos(yaw_rad) -sin(yaw_rad) 0;
              sin(yaw_rad)  cos(yaw_rad) 0;
              0             0            1];
    R_pitch = [ 1  0              0;
                0  cos(pitch_rad) -sin(pitch_rad);
                0  sin(pitch_rad)  cos(pitch_rad)];
    R_cam = R_yaw * R_pitch;

    % ==================== VISION PIPELINE DETECTION ====================
    camera_forward = R_yaw * [0; -1; 0];

    for i = 1:numel(all_kfs_ids)
        kb     = forest_blocks(all_kfs_ids(i));
        kt     = kfs_types(i);
        kfs_id = all_kfs_ids(i);

        target_pt   = [kb.x, kb.y, kb.h + KFS_SIZE/2];
        dist_to_kfs = norm(camera_pos(1:2) - target_pt(1:2));

        if dist_to_kfs <= DETECTION_RANGE
            to_kfs      = target_pt(1:2)' - camera_pos(1:2)';
            to_kfs_norm = to_kfs / norm(to_kfs);
            angle       = acosd(dot(camera_forward(1:2), to_kfs_norm));

            if angle <= DETECTION_FOV/2
                is_visible = ~isOccluded(camera_pos, target_pt, forest_blocks, BLOCK_SIZE, kb.id);

                if is_visible
                    % HSV detection based on color (blue)
                    kc      = kfs_colors{i};
                    is_blue = all(abs(kc - COLOR_KFS_BASE_BLUE) < 1e-3);
                    hsv_detected = is_blue;

                    if hsv_detected
                        ml_confidence = 0.6 + 0.3 * (1 - dist_to_kfs/DETECTION_RANGE);
                        ml_confidence = max(0.5, min(0.95, ml_confidence));

                        is_real = (kt == "R2");
                        if is_real
                            simulated_score = CONF_THRESHOLD - 0.1; %#ok<NASGU>
                        else
                            simulated_score = CONF_THRESHOLD + 0.1; %#ok<NASGU>
                        end

                        if ~detection_timers.isKey(kfs_id)
                            detection_timers(kfs_id) = tic;
                        end

                        if toc(detection_timers(kfs_id)) >= MIN_DETECTION_TIME
                            if ~detected_kfs.isKey(kfs_id)
                                detected_kfs(kfs_id) = struct('type',kt,'confidence',ml_confidence);

                                log_entry = struct(...
                                    'time',       toc(mission_start_time), ...
                                    'kfs_id',     kfs_id, ...
                                    'type',       char(kt), ...
                                    'confidence', ml_confidence, ...
                                    'pos',        robot_pos);
                                detection_log(end+1) = log_entry;

                                if kt == "R2"
                                    stats.r2_found = stats.r2_found + 1;
                                elseif kt == "R1"
                                    stats.r1_found = stats.r1_found + 1;
                                else
                                    stats.fake_found = stats.fake_found + 1;
                                end

                                fprintf('[VISION] Detected %s at Block %d (Conf: %.2f, Dist: %.0fmm)\n', ...
                                        kt, kfs_id, ml_confidence, dist_to_kfs);
                            end
                        end
                    else
                        if detection_timers.isKey(kfs_id)
                            detection_timers.remove(kfs_id);
                        end
                    end
                else
                    if detection_timers.isKey(kfs_id)
                        detection_timers.remove(kfs_id);
                    end
                end
            else
                if detection_timers.isKey(kfs_id)
                    detection_timers.remove(kfs_id);
                end
            end
        else
            if detection_timers.isKey(kfs_id)
                detection_timers.remove(kfs_id);
            end
        end
    end

    % Stats update
    stats.total_distance = stats.total_distance + norm(robot_pos - last_pos);
    stats.mission_time   = toc(mission_start_time);
    last_pos = robot_pos;

    % ==================== OVERVIEW VIEW (LEFT) ====================
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

    % KFS cubes
    for i = 1:numel(all_kfs_ids)
        kb     = forest_blocks(all_kfs_ids(i));
        kfs_id = all_kfs_ids(i);

        kc = kfs_colors{i};

        if detected_kfs.isKey(kfs_id)
            edge_c = [0 1 0]; edge_w = 3;
        else
            edge_c = 'k'; edge_w = 1;
        end

        drawBox(ax_main, kb.x-KFS_SIZE/2, kb.y-KFS_SIZE/2, kb.h, ...
                KFS_SIZE, KFS_SIZE, KFS_SIZE, kc, 0.5, edge_c, edge_w);

        if i <= numel(kfs_images) && ~isempty(kfs_images{i})
            img = kfs_images{i};
            drawTexturedKFS(ax_main, kb, KFS_SIZE, img);
        end
    end

    % Robot
    drawBox(ax_main, robot_pos(1)-R2_W/2, robot_pos(2)-R2_L/2, robot_pos(3), ...
            R2_W, R2_L, R2_H_GROUND, COLOR_ROBOT, 0.8, 'k', 2);

    arrow_len = 500;
    arrow_end = robot_pos + [cos(deg2rad(robot_yaw-90))*arrow_len, ...
                             sin(deg2rad(robot_yaw-90))*arrow_len, 0];
    quiver3(robot_pos(1), robot_pos(2), robot_pos(3)+R2_H_GROUND/2, ...
            arrow_end(1)-robot_pos(1), arrow_end(2)-robot_pos(2), 0, ...
            'r','LineWidth',3,'MaxHeadSize',2);

    % Detection range circle
    theta = linspace(0, 2*pi, 50);
    det_x = robot_pos(1) + DETECTION_RANGE * cos(theta);
    det_y = robot_pos(2) + DETECTION_RANGE * sin(theta);
    plot3(det_x, det_y, ones(size(theta))*robot_pos(3), 'c--', 'LineWidth', 1);

    % Path
    if ~isempty(path_history)
        plot3(path_history(:,1), path_history(:,2), path_history(:,3), ...
              'g-', 'LineWidth', 2);
    end

    title(sprintf('Mission: %.1fs | Dist: %.1fm | Robot Yaw: %.0f° | Cam Offset: %.0f°', ...
          stats.mission_time, stats.total_distance/1000, robot_yaw, camera_yaw_offset), ...
          'Color', 'white');

    % ==================== CAMERA VIEW (RIGHT) ====================
    set(fig, 'CurrentAxes', ax_cam);
    cla(ax_cam); hold on; axis equal; axis off;
    view(3);

    camproj(ax_cam, 'perspective');

    cam_target = camera_pos + (R_cam * [0; -1000; 0])';
    cam_up     = (R_cam * [0; 0; 1])';

    campos(ax_cam,   camera_pos);
    camtarget(ax_cam, cam_target);
    camup(ax_cam,     cam_up);
    camva(ax_cam,     CAMERA_FOV_H);

    axis_range = CAMERA_RANGE * 1.2;
    xlim([camera_pos(1)-axis_range, camera_pos(1)+axis_range]);
    ylim([camera_pos(2)-axis_range, camera_pos(2)+axis_range]);
    zlim([0, ARENA_Z]);

    set(ax_cam, 'Color', [0.5 0.6 0.8]);

    % Ground
    patch([0 ARENA_X ARENA_X 0],[0 0 ARENA_Y ARENA_Y],[0 0 0 0], ...
          COLOR_PATHWAY,'FaceAlpha',0.5,'EdgeColor','none');

    % Blocks
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

    % KFS in camera view
    for i = 1:numel(all_kfs_ids)
        kb     = forest_blocks(all_kfs_ids(i));
        kfs_id = all_kfs_ids(i);
        kc     = kfs_colors{i};

        dist = norm(camera_pos(1:2) - [kb.x, kb.y]);
        if dist > CAMERA_RANGE * 1.5, continue; end

        target_pt  = [kb.x, kb.y, kb.h + KFS_SIZE/2];
        is_visible = ~isOccluded(camera_pos, target_pt, forest_blocks, BLOCK_SIZE, kb.id);

        if is_visible && dist <= DETECTION_RANGE
            to_kfs      = target_pt(1:2)' - camera_pos(1:2)';
            to_kfs_norm = to_kfs / norm(to_kfs);
            camera_forward_2d = (R_yaw * [0; -1; 0]);
            angle = acosd(dot(camera_forward_2d(1:2), to_kfs_norm));

            if angle <= DETECTION_FOV/2
                kfs_in_view = kfs_in_view + 1;
                edge_col   = [0 1 0];
                edge_width = 3;
            else
                edge_col   = [0.5 0.5 0.5];
                edge_width = 1;
            end
        else
            edge_col   = [0.3 0.3 0.3];
            edge_width = 0.5;
        end

        drawBox(ax_cam, kb.x-KFS_SIZE/2, kb.y-KFS_SIZE/2, kb.h, ...
                KFS_SIZE, KFS_SIZE, KFS_SIZE, kc, 0.4, edge_col, edge_width);

        if i <= numel(kfs_images) && ~isempty(kfs_images{i})
            img = kfs_images{i};
            drawTexturedKFS(ax_cam, kb, KFS_SIZE, img);
        end
    end

    % HUD
    text(0.02, 0.98, sprintf('R2: %d/%d | R1: %d/%d | FAKE: %d/%d | In View: %d', ...
         stats.r2_found, total_r2_real, ...
         stats.r1_found, total_r1, ...
         stats.fake_found, total_fake, ...
         kfs_in_view), ...
         'Units','normalized', 'Color',[0 1 0], 'FontSize',11, 'FontWeight','bold', ...
         'BackgroundColor',[0 0 0 0.7], 'VerticalAlignment','top');

    auto_tilt_str = '';
    if auto_tilt_enabled
        auto_tilt_str = ' [AUTO-TILT ON]';
    end
    text(0.02, 0.90, sprintf('Pitch: %.0f° | Yaw: %.0f°%s', ...
         camera_pitch, camera_yaw, auto_tilt_str), ...
         'Units','normalized', 'Color','white', 'FontSize',10, ...
         'BackgroundColor',[0 0 0 0.7], 'VerticalAlignment','top');

    if recording_path
        text(0.98, 0.98, '● REC', 'Units','normalized', 'Color','red', ...
             'FontSize',14, 'FontWeight','bold', 'HorizontalAlignment','right', ...
             'BackgroundColor',[0 0 0 0.7], 'VerticalAlignment','top');
    end

    text(0.5, 0.5, '+', 'Units','normalized', 'Color',[0 1 0], ...
         'FontSize',24, 'HorizontalAlignment','center', 'VerticalAlignment','middle');

    title('CAMERA FEED (Vision Pipeline Active)', 'Color', 'white', 'FontSize', 14);

    % Help
    if show_help
        axes(ax_main);
        help_text = sprintf([...
            ' CONTROLS:\n' ...
            ' W/S  : Fwd/Back\n' ...
            ' A/D  : Strafe\n' ...
            ' Q/E  : Rotate Bot\n' ...
            ' I/K  : Pan Camera\n' ...
            ' Z/X  : Pitch\n' ...
            ' V    : Auto-Tilt\n' ...
            ' SPC  : Climb\n' ...
            ' C    : Down\n' ...
            ' R    : Reset\n' ...
            ' T    : Randomize\n' ...
            ' P    : Record\n' ...
            ' L    : Export']);
        text(0.02, 0.98, help_text, 'Units','normalized', ...
             'VerticalAlignment','top', 'Color','yellow', ...
             'BackgroundColor',[0 0 0 0.8], 'FontName','FixedWidth', 'FontSize', 9);
    end

    drawnow;
end

%% ======================= NESTED FUNCTIONS ==========================

function [real_imgs, fake_imgs] = loadOracleBoneImages(real_path, fake_path)
    real_imgs = {};
    fake_imgs = {};

    if ~isempty(real_path) && exist(real_path, 'dir')
        files = dir(fullfile(real_path, '*.png'));
        files = [files; dir(fullfile(real_path, '*.jpg'))];
        for f = 1:numel(files)
            try
                img = imread(fullfile(real_path, files(f).name));
                if size(img,3)==1, img = repmat(img,[1 1 3]); end
                real_imgs{end+1} = img;
            catch
                fprintf('[WARN] Failed to load: %s\n', files(f).name);
            end
        end
    end

    if ~isempty(fake_path) && exist(fake_path, 'dir')
        files = dir(fullfile(fake_path, '*.png'));
        files = [files; dir(fullfile(fake_path, '*.jpg'))];
        for f = 1:numel(files)
            try
                img = imread(fullfile(fake_path, files(f).name));
                if size(img,3)==1, img = repmat(img,[1 1 3]); end
                fake_imgs{end+1} = img;
            catch
                fprintf('[WARN] Failed to load: %s\n', files(f).name);
            end
        end
    end

    fprintf('[OK] Loaded %d REAL, %d FAKE oracle bone images\n', ...
            numel(real_imgs), numel(fake_imgs));
end

function randomizeKFS()
    % R1 only on edges/corners
    edge_ids  = [];
    inner_ids = [];
    for k = 1:numel(forest_blocks)
        b = forest_blocks(k);
        if b.row == 1 || b.row == FOREST_ROWS || ...
           b.col == 1 || b.col == FOREST_COLS
            edge_ids(end+1) = b.id; %#ok<AGROW>
        else
            inner_ids(end+1) = b.id; %#ok<AGROW>
        end
    end

    if numel(edge_ids) < total_r1
        error('Not enough edge blocks for R1 placement.');
    end

    edge_ids_perm = edge_ids(randperm(numel(edge_ids)));
    r1_ids        = edge_ids_perm(1:total_r1);

    remaining_ids  = setdiff(1:numel(forest_blocks), r1_ids);
    remaining_perm = remaining_ids(randperm(numel(remaining_ids)));

    r2_real_ids = remaining_perm(1:total_r2_real);
    fake_ids    = remaining_perm(total_r2_real+1 : total_r2_real+total_fake);

    all_kfs_ids = [r1_ids, r2_real_ids, fake_ids];
    kfs_types   = [repmat("R1",1,total_r1), ...
                   repmat("R2",1,total_r2_real), ...
                   "FAKE"];

    kfs_images = cell(1, numel(all_kfs_ids));
    for k = 1:numel(all_kfs_ids)
        if kfs_types(k) == "R1"
            kfs_images{k} = r1_images{randi(numel(r1_images))};
        elseif kfs_types(k) == "R2"
            kfs_images{k} = r2_real_images{randi(numel(r2_real_images))};
        else
            kfs_images{k} = r2_fake_images{randi(numel(r2_fake_images))};
        end
    end

    kfs_colors = cell(1, numel(all_kfs_ids));
    for k = 1:numel(all_kfs_ids)
        if rand < 0.5
            kfs_colors{k} = COLOR_KFS_BASE_RED;
        else
            kfs_colors{k} = COLOR_KFS_BASE_BLUE;
        end
    end

    detected_kfs     = containers.Map('KeyType','double','ValueType','any');
    detection_timers = containers.Map('KeyType','double','ValueType','uint64');
    stats.r2_found   = 0;
    stats.r1_found   = 0;
    stats.fake_found = 0;
    detection_log    = struct('time',{},'kfs_id',{},'type',{},'confidence',{},'pos',{});

    fprintf('[RANDOMIZER] New layout | R2 real at: %s | R1 at: %s | Fake at: %s\n', ...
            mat2str(r2_real_ids), mat2str(r1_ids), mat2str(fake_ids));
end

function pitch = autoAdjustPitch(cam_pos, cam_yaw)
    persistent target_pitch;
    if isempty(target_pitch), target_pitch = 15; end

    cam_fwd = [cos(deg2rad(cam_yaw-90)), sin(deg2rad(cam_yaw-90)), 0];

    min_dist = inf;
    best_kfs = [];

    for i = 1:numel(all_kfs_ids)
        kb = forest_blocks(all_kfs_ids(i));
        kfs_center = [kb.x, kb.y, kb.h + KFS_SIZE/2];

        to_kfs = kfs_center - cam_pos;
        dist   = norm(to_kfs);

        if dist > DETECTION_RANGE, continue; end

        if dot(to_kfs(1:2), cam_fwd(1:2)) > 0
            if dist < min_dist
                min_dist = dist;
                best_kfs = kfs_center;
            end
        end
    end

    if ~isempty(best_kfs)
        delta_z         = best_kfs(3) - cam_pos(3);
        horizontal_dist = norm(best_kfs(1:2) - cam_pos(1:2));
        target_pitch    = atand(delta_z / horizontal_dist);
        target_pitch    = max(-45, min(45, target_pitch));
    end

    pitch = camera_pitch + (target_pitch - camera_pitch) * 0.1;
end

function drawTexturedKFS(ax, kb, KFS_SIZE, img)
    texSize = 200; % mm
    xi0 = kb.x - texSize/2;
    xi1 = kb.x + texSize/2;
    yi0 = kb.y - texSize/2;
    yi1 = kb.y + texSize/2;

    x0 = kb.x - KFS_SIZE/2;
    y0 = kb.y - KFS_SIZE/2;
    z0 = kb.h;
    x1 = x0 + KFS_SIZE;
    y1 = y0 + KFS_SIZE;
    z1 = z0 + KFS_SIZE;

    function drawFace(X,Y,Z)
        surf(ax, X, Y, Z, 'CData', img, 'FaceColor','texturemap', ...
             'EdgeColor','none','FaceAlpha',0.95);
    end

    % Top
    X = [xi0 xi1; xi0 xi1];
    Y = [yi0 yi0; yi1 yi1];
    Z = [z1  z1;  z1  z1];
    drawFace(X,Y,Z);

    % Front (+Y)
    X = [xi0 xi1; xi0 xi1];
    Y = [y1  y1;  y1  y1];
    Z = [z0  z0;  z1  z1];
    drawFace(X,Y,Z);

    % Back (-Y)
    X = [xi1 xi0; xi1 xi0];
    Y = [y0  y0;  y0  y0];
    Z = [z0  z0;  z1  z1];
    drawFace(X,Y,Z);

    % Left (-X)
    X = [x0  x0;  x0  x0];
    Y = [yi0 yi1; yi0 yi1];
    Z = [z0  z0;  z1  z1];
    drawFace(X,Y,Z);

    % Right (+X)
    X = [x1  x1;  x1  x1];
    Y = [yi1 yi0; yi1 yi0];
    Z = [z0  z0;  z1  z1];
    drawFace(X,Y,Z);
end

function keyHandler(~, event)
    switch event.Key
        case 'escape'
            running = false;

        case 'h'
            show_help = ~show_help;

        case 'v'
            auto_tilt_enabled = ~auto_tilt_enabled;
            fprintf('[AUTO-TILT] %s\n', string(auto_tilt_enabled));

        case 'p'
            recording_path = ~recording_path;
            fprintf('[PATH] Recording %s\n', string(recording_path));

        case 't'
            randomizeKFS();

        case 'l'
            exportTrajectory();

        case 'r'
            current_block_id = INITIAL_BLOCK_ID;
            nb = forest_blocks(INITIAL_BLOCK_ID);
            robot_pos = [nb.x, nb.y, 0];
            robot_yaw = 0;
            camera_yaw_offset = 0;
            camera_pitch = 15;
            path_history = [];
            stats.total_distance = 0;
            mission_start_time = tic;
            fprintf('[RESET] Robot reset\n');

        case 'w'
            [new_pos, valid] = moveRobot(robot_pos, robot_yaw, MOVE_SPEED, 0);
            if valid, robot_pos = new_pos; updatePathHistory(); end

        case 's'
            [new_pos, valid] = moveRobot(robot_pos, robot_yaw, -MOVE_SPEED, 0);
            if valid, robot_pos = new_pos; updatePathHistory(); end

        case 'a'
            [new_pos, valid] = moveRobot(robot_pos, robot_yaw, 0, MOVE_SPEED);
            if valid, robot_pos = new_pos; updatePathHistory(); end

        case 'd'
            [new_pos, valid] = moveRobot(robot_pos, robot_yaw, 0, -MOVE_SPEED);
            if valid, robot_pos = new_pos; updatePathHistory(); end

        case 'space'
            for k = 1:numel(forest_blocks)
                b = forest_blocks(k);
                if abs(robot_pos(1) - b.x) < BLOCK_SIZE/2 && ...
                   abs(robot_pos(2) - b.y) < BLOCK_SIZE/2
                    robot_pos(3) = b.h;
                    current_block_id = b.id;
                    fprintf('[CLIMB] Block %d (H=%d)\n', b.id, b.h);
                    break;
                end
            end

        case 'c'
            robot_pos(3) = 0;
            fprintf('[DESCEND] Ground level\n');

        case 'q'
            robot_yaw = robot_yaw + ROT_SPEED;

        case 'e'
            robot_yaw = robot_yaw - ROT_SPEED;

        case 'i'
            camera_yaw_offset = camera_yaw_offset + YAW_STEP;
            fprintf('[CAMERA] Pan: %.0f°\n', camera_yaw_offset);

        case 'k'
            camera_yaw_offset = camera_yaw_offset - YAW_STEP;
            fprintf('[CAMERA] Pan: %.0f°\n', camera_yaw_offset);

        case 'z'
            if ~auto_tilt_enabled
                camera_pitch = max(-45, camera_pitch - PITCH_STEP);
                fprintf('[CAMERA] Pitch: %.0f°\n', camera_pitch);
            end

        case 'x'
            if ~auto_tilt_enabled
                camera_pitch = min(45, camera_pitch + PITCH_STEP);
                fprintf('[CAMERA] Pitch: %.0f°\n', camera_pitch);
            end
    end
end

function [new_pos, valid] = moveRobot(pos, yaw, fwd, side)
    yaw_rad = deg2rad(yaw - 90);

    dx = fwd * cos(yaw_rad) - side * sin(yaw_rad);
    dy = fwd * sin(yaw_rad) + side * cos(yaw_rad);

    new_pos = pos + [dx, dy, 0];

    % Arena boundary check
    if new_pos(1) < R2_W/2 || new_pos(1) > ARENA_X - R2_W/2 || ...
       new_pos(2) < R2_L/2 || new_pos(2) > ARENA_Y - R2_L/2
        valid = false;
        fprintf('[COLLISION] Arena boundary\n');
        return;
    end

    % Forest rectangle (for elevated vs off-forest)
    forest_min_x = forest_x_min;
    forest_max_x = forest_x_min + FOREST_COLS * BLOCK_SIZE;
    forest_min_y = forest_y_min;
    forest_max_y = forest_y_min + FOREST_ROWS * BLOCK_SIZE;

    % Determine which block cell (if any) the new position is over
    on_block_id = 0;
    for k = 1:numel(forest_blocks)
        b = forest_blocks(k);
        if abs(new_pos(1) - b.x) < BLOCK_SIZE/2 && ...
           abs(new_pos(2) - b.y) < BLOCK_SIZE/2
            on_block_id = b.id;
            break;
        end
    end

    if pos(3) <= 10
        % ================= GROUND MOVEMENT =================
        % Allow free movement anywhere over forest area at z=0
        new_pos(3) = 0;
        valid = true;

        % Optional: forbid going outside forest rectangle on ground
        % (comment out if you want totally free ground movement)
        if new_pos(1) < forest_min_x || new_pos(1) > forest_max_x || ...
           new_pos(2) < forest_min_y || new_pos(2) > forest_max_y
            valid = false;
            fprintf('[COLLISION] Off forest area (ground)\n');
            return;
        end

        if on_block_id ~= 0
            current_block_id = on_block_id;
        end
        return;

    else
        % ================= ELEVATED MOVEMENT =================
        % Only forbid leaving the forest region entirely
        if new_pos(1) < forest_min_x || new_pos(1) > forest_max_x || ...
           new_pos(2) < forest_min_y || new_pos(2) > forest_max_y
            valid = false;
            fprintf('[COLLISION] Off blocks (elevated, outside forest)\n');
            return;
        end

        % Stay at current height by default
        new_pos(3) = pos(3);
        valid = true;

        % If over a block cell, snap to that block's height
        if on_block_id ~= 0
            b = forest_blocks(on_block_id);
            new_pos(3)      = b.h;
            current_block_id = b.id;
        end

        return;
    end
end


function updatePathHistory()
    if recording_path
        path_history(end+1, :) = robot_pos;
    end
end

function exportTrajectory()
    timestamp = datestr(now, 'yyyymmdd_HHMMSS');

    if ~isempty(path_history)
        csv_file = sprintf('trajectory_%s.csv', timestamp);
        csvwrite(csv_file, path_history);
        fprintf('[EXPORT] Path: %s\n', csv_file);
    end

    mat_file = sprintf('mission_data_%s.mat', timestamp);
    save(mat_file, 'path_history', 'detection_log', 'stats', ...
         'all_kfs_ids', 'kfs_types', 'kfs_colors');
    fprintf('[EXPORT] Data: %s\n', mat_file);

    fprintf('\n=== MISSION SUMMARY ===\n');
    fprintf('Duration: %.1fs\n', stats.mission_time);
    fprintf('Distance: %.2fm\n', stats.total_distance/1000);
    fprintf('R2: %d/%d | R1: %d/%d | Fake: %d/%d\n', ...
            stats.r2_found, total_r2_real, ...
            stats.r1_found, total_r1, ...
            stats.fake_found, total_fake);
    fprintf('Detections: %d\n', numel(detection_log));
    fprintf('======================\n\n');
end

end % main function

%% ======================= LOCAL HELPER FUNCTIONS =======================
function drawBox(ax, x, y, z, w, l, h, col, alph, edge_c, edge_w)
    V = [x y z; x+w y z; x+w y+l z; x y+l z;
         x y z+h; x+w y z+h; x+w y+l z+h; x y+l z+h];
    F = [1 2 6 5; 2 3 7 6; 3 4 8 7; 4 1 5 8; 1 2 3 4; 5 6 7 8];
    patch(ax, 'Vertices',V,'Faces',F,'FaceColor',col,'FaceAlpha',alph, ...
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
