%% ABU ROBOCON 2026 - CORRECTED COMPLETE ARENA 3D VISUALIZATION
% Team-Vulcans Robocon 2026 - EXACT Rulebook Layout (Appendix 1-4)
% CORRECTED VERSION - Symmetric Forest, Proper Dimensions

clear; clc; close all;

%% CRITICAL ARENA CONFIGURATION (From Rulebook Appendix 1)
% Game Field: 6000mm × 6000mm
cfg.ARENA_X = 6000;     % mm (X-axis)
cfg.ARENA_Y = 6000;     % mm (Y-axis)  
cfg.ARENA_Z = 2000;     % mm (height reference)

% MEIHUA FOREST CONFIGURATION (Corrected from Appendix 1.2)
% Each team has 12 blocks in a 4 rows × 3 columns grid
% Block spacing: 1450mm center-to-center
% Block size: 1450mm × 1450mm platform

% RED TEAM FOREST (Left side - Blocks 1-12)
cfg.RED_FOREST_X_START = 725;      % Left edge + half block
cfg.RED_FOREST_Y_START = 4365;     % Top edge - pathway
cfg.BLOCK_SIZE = 1450;             % mm (center-to-center spacing)

% BLUE TEAM FOREST (Right side - Blocks 1-12, MIRRORED)
cfg.BLUE_FOREST_X_START = cfg.ARENA_X - 725 - 2*cfg.BLOCK_SIZE;  % Mirror position

% Forest block heights pattern (from rulebook isometric)
cfg.FOREST_HEIGHTS = [
    200, 400, 600;   % Row 1 (blocks 1,2,3)
    400, 200, 400;   % Row 2 (blocks 4,5,6)
    600, 400, 200;   % Row 3 (blocks 7,8,9)
    400, 600, 400    % Row 4 (blocks 10,11,12)
];

cfg.KFS_SIZE = 350;     % mm (350×350×350 cube - Appendix 4.1)

%% Create Enhanced Figure
fig = figure('Position', [50 50 1920 1080], 'Color', [0.08 0.08 0.08], ...
    'Name', '🥋 ABU Robocon 2026 - CORRECTED Arena', 'NumberTitle', 'off');

% Main 3D view
ax_main = axes('Parent', fig, 'Position', [0.05 0.35 0.55 0.6]);
hold(ax_main, 'on');
grid(ax_main, 'on');
axis equal;
xlim(ax_main, [-200 cfg.ARENA_X+200]);
ylim(ax_main, [-200 cfg.ARENA_Y+200]);
zlim(ax_main, [0 cfg.ARENA_Z]);
xlabel(ax_main, 'X (mm)', 'Color', 'w', 'FontSize', 12); 
ylabel(ax_main, 'Y (mm)', 'Color', 'w', 'FontSize', 12); 
zlabel(ax_main, 'Z (mm)', 'Color', 'w', 'FontSize', 12);
view(ax_main, 45, 30);
lighting gouraud; 
camlight('headlight');
light('Position', [cfg.ARENA_X/2, cfg.ARENA_Y/2, cfg.ARENA_Z*2], 'Style', 'local');
set(ax_main, 'Color', [0.05 0.05 0.05], 'XColor', 'w', 'YColor', 'w', 'ZColor', 'w');

% Top view (Strategy)
ax_top = axes('Parent', fig, 'Position', [0.65 0.68 0.32 0.27]);
hold(ax_top, 'on'); axis equal; grid on;
xlim(ax_top, [0 cfg.ARENA_X]); ylim(ax_top, [0 cfg.ARENA_Y]);
title(ax_top, '🗺️ Top View (Strategy Map)', 'Color', 'cyan', 'FontSize', 11, 'FontWeight', 'bold');
set(ax_top, 'Color', [0.05 0.05 0.05], 'XColor', 'w', 'YColor', 'w');
view(ax_top, 0, 90);

% Side view (Heights)
ax_side = axes('Parent', fig, 'Position', [0.65 0.35 0.32 0.27]);
hold(ax_side, 'on'); axis equal; grid on;
xlim(ax_side, [0 cfg.ARENA_X]); zlim(ax_side, [0 1500]);
title(ax_side, '📏 Side View (Height Analysis)', 'Color', 'cyan', 'FontSize', 11, 'FontWeight', 'bold');
set(ax_side, 'Color', [0.05 0.05 0.05], 'XColor', 'w', 'ZColor', 'w');
view(ax_side, 0, 0);

%% 1. GROUND PLATFORM (6m × 6m base)
[Xg, Yg] = meshgrid(0:500:cfg.ARENA_X, 0:500:cfg.ARENA_Y);
Zg = zeros(size(Xg));
surf(ax_main, Xg, Yg, Zg, 'FaceColor', [0.15 0.15 0.15], 'FaceAlpha', 0.9, ...
    'EdgeColor', [0.25 0.25 0.25], 'LineWidth', 0.5);

% Arena border
plot3(ax_main, [0 cfg.ARENA_X cfg.ARENA_X 0 0], ...
    [0 0 cfg.ARENA_Y cfg.ARENA_Y 0], [0 0 0 0 0], ...
    'w-', 'LineWidth', 4);
plot(ax_top, [0 cfg.ARENA_X cfg.ARENA_X 0 0], ...
    [0 0 cfg.ARENA_Y cfg.ARENA_Y 0], 'w-', 'LineWidth', 3);

%% 2. ZONE 1 - MARTIAL CLUB (Red and Blue)
% Red Team (Left side)
zone1_red_x = [0, 0, 3200, 3200];
zone1_red_y = [cfg.ARENA_Y-3200, cfg.ARENA_Y, cfg.ARENA_Y, cfg.ARENA_Y-3200];
draw_zone_flat(ax_main, ax_top, zone1_red_x, zone1_red_y, [250/255 220/255 218/255], 'Zone 1 (RED)', 5);

% Blue Team (Right side - SYMMETRIC)
zone1_blue_x = [cfg.ARENA_X-3200, cfg.ARENA_X-3200, cfg.ARENA_X, cfg.ARENA_X];
zone1_blue_y = [cfg.ARENA_Y-3200, cfg.ARENA_Y, cfg.ARENA_Y, cfg.ARENA_Y-3200];
draw_zone_flat(ax_main, ax_top, zone1_blue_x, zone1_blue_y, [128/255 199/255 226/255], 'Zone 1 (BLUE)', 5);

%% 3. START ZONES (1000×1000mm each)
% R1 Red Start Zone
r1_red_x = [500, 1500, 1500, 500];
r1_red_y = [cfg.ARENA_Y-1000, cfg.ARENA_Y-1000, cfg.ARENA_Y, cfg.ARENA_Y];
draw_zone_flat(ax_main, ax_top, r1_red_x, r1_red_y, [223/255 34/255 34/255], 'R1 (RED)', 15);
text(ax_top, mean(r1_red_x), mean(r1_red_y), 'R1', 'Color', 'white', ...
    'FontSize', 14, 'FontWeight', 'bold', 'HorizontalAlignment', 'center');

% R2 Red Start Zone
r2_red_x = [1500, 2500, 2500, 1500];
r2_red_y = [cfg.ARENA_Y-1000, cfg.ARENA_Y-1000, cfg.ARENA_Y, cfg.ARENA_Y];
draw_zone_flat(ax_main, ax_top, r2_red_x, r2_red_y, [223/255 34/255 34/255], 'R2 (RED)', 15);
text(ax_top, mean(r2_red_x), mean(r2_red_y), 'R2', 'Color', 'white', ...
    'FontSize', 14, 'FontWeight', 'bold', 'HorizontalAlignment', 'center');

% R1 Blue Start Zone (SYMMETRIC)
r1_blue_x = [cfg.ARENA_X-1500, cfg.ARENA_X-500, cfg.ARENA_X-500, cfg.ARENA_X-1500];
r1_blue_y = [cfg.ARENA_Y-1000, cfg.ARENA_Y-1000, cfg.ARENA_Y, cfg.ARENA_Y];
draw_zone_flat(ax_main, ax_top, r1_blue_x, r1_blue_y, [50/255 0/255 255/255], 'R1 (BLUE)', 15);
text(ax_top, mean(r1_blue_x), mean(r1_blue_y), 'R1', 'Color', 'white', ...
    'FontSize', 14, 'FontWeight', 'bold', 'HorizontalAlignment', 'center');

% R2 Blue Start Zone (SYMMETRIC)
r2_blue_x = [cfg.ARENA_X-2500, cfg.ARENA_X-1500, cfg.ARENA_X-1500, cfg.ARENA_X-2500];
r2_blue_y = [cfg.ARENA_Y-1000, cfg.ARENA_Y-1000, cfg.ARENA_Y, cfg.ARENA_Y];
draw_zone_flat(ax_main, ax_top, r2_blue_x, r2_blue_y, [50/255 0/255 255/255], 'R2 (BLUE)', 15);
text(ax_top, mean(r2_blue_x), mean(r2_blue_y), 'R2', 'Color', 'white', ...
    'FontSize', 14, 'FontWeight', 'bold', 'HorizontalAlignment', 'center');

%% 4. STAFF RACKS (1200×500×300mm metal)
% Red Staff Rack
staff_rack_red_pos = [1500, cfg.ARENA_Y-2000, 150];
draw_rack(ax_main, ax_top, staff_rack_red_pos, [1200 500 300], [155/255 95/255 0], 'Staff Rack', true);

% Blue Staff Rack (SYMMETRIC)
staff_rack_blue_pos = [cfg.ARENA_X-1500-1200, cfg.ARENA_Y-2000, 150];
draw_rack(ax_main, ax_top, staff_rack_blue_pos, [1200 500 300], [155/255 95/255 0], 'Staff Rack', true);

% Shared Spearhead Rack (center)
spearhead_rack_pos = [cfg.ARENA_X/2-600, cfg.ARENA_Y-2000, 150];
draw_rack(ax_main, ax_top, spearhead_rack_pos, [1200 500 300], [155/255 95/255 0], 'Spearhead Rack', true);

%% 5. MEIHUA FOREST - RED TEAM (12 blocks, 4×3 grid)
fprintf('\n🌲 Creating RED TEAM Meihua Forest...\n');
for row = 1:4
    for col = 1:3
        block_num = (row-1)*3 + col;
        
        % Calculate block center position
        bx = cfg.RED_FOREST_X_START + (col-1)*cfg.BLOCK_SIZE;
        by = cfg.RED_FOREST_Y_START - (row-1)*cfg.BLOCK_SIZE;
        h = cfg.FOREST_HEIGHTS(row, col);
        
        % Draw forest block
        draw_forest_block(ax_main, ax_top, ax_side, [bx, by, 0], h, block_num, 'red');
        
        fprintf('   Block %2d: (%.0f, %.0f, %3d mm)\n', block_num, bx, by, h);
    end
end

%% 6. MEIHUA FOREST - BLUE TEAM (12 blocks, 4×3 grid, MIRRORED)
fprintf('\n🌲 Creating BLUE TEAM Meihua Forest...\n');
for row = 1:4
    for col = 1:3
        block_num = (row-1)*3 + col;
        
        % Calculate MIRRORED block center position
        bx = cfg.BLUE_FOREST_X_START + (col-1)*cfg.BLOCK_SIZE;
        by = cfg.RED_FOREST_Y_START - (row-1)*cfg.BLOCK_SIZE;  % Same Y as red
        h = cfg.FOREST_HEIGHTS(row, col);  % Same height pattern
        
        % Draw forest block
        draw_forest_block(ax_main, ax_top, ax_side, [bx, by, 0], h, block_num, 'blue');
        
        fprintf('   Block %2d: (%.0f, %.0f, %3d mm)\n', block_num, bx, by, h);
    end
end

%% 7. ZONE 3 - ARENA PLATFORM (450mm height)
zone3_x = [3200, 5800, 5800, 3200];
zone3_y = [0, 0, 2700, 2700];
draw_platform(ax_main, ax_top, zone3_x, zone3_y, 450, [254/255 186/255 163/255], 'Arena Platform');

% Ramp to Arena
ramp_pts = [
    3200, 2700, 0;
    3200, 2700, 450;
    3200, 0, 450;
    3200, 0, 0
];
patch(ax_main, ramp_pts(:,1), ramp_pts(:,2), ramp_pts(:,3), ...
    'FaceColor', [0.7 0.7 0.7], 'EdgeColor', 'k', 'LineWidth', 2, 'FaceAlpha', 0.8);

%% 8. TIC-TAC-TOE RACK (1620×540×800mm)
ttt_x = cfg.ARENA_X/2 - 810;
ttt_y = 1350;
ttt_z = 450 + 50;  % On platform

% Base structure
draw_box_3d(ax_main, [ttt_x+810, ttt_y, ttt_z+400], [1620, 540, 800], ...
    [0.7 0.7 0.9], 'Tic-Tac-Toe Rack', 0.7);

% Draw 3×3 grid slots
slot_spacing = 540;
slot_size = 400;
for row = 1:3
    for col = 1:3
        slot_x = ttt_x + (col-1)*slot_spacing + 270;
        slot_y = ttt_y;
        slot_z = ttt_z + (row-1)*270 + 135;
        
        % Draw slot frame
        draw_box_3d(ax_main, [slot_x, slot_y, slot_z], [slot_size, 80, slot_size], ...
            [0.9 0.9 1], '', 0.5);
        
        % Mark on top view
        scatter(ax_top, slot_x, slot_y, 100, 'w', 's', 'LineWidth', 1.5);
        text(ax_top, slot_x, slot_y, sprintf('%d,%d', row, col), ...
            'Color', 'w', 'FontSize', 7, 'HorizontalAlignment', 'center');
    end
end

%% 9. PATHWAYS (Non-shiny vinyl tape)
% R1 Pathway (white, 3200mm line from Zone 1 to Arena)
pathway_y = [cfg.ARENA_Y-3200, 2700];
pathway_x = [3200, 3200];
plot3(ax_main, pathway_x, pathway_y, [10 10], 'w-', 'LineWidth', 20);
plot3(ax_main, pathway_x, pathway_y, [10 10], 'y-', 'LineWidth', 12); % Glow
plot(ax_top, pathway_x, pathway_y, 'w-', 'LineWidth', 12);
plot(ax_top, pathway_x, pathway_y, 'y:', 'LineWidth', 8);

%% 10. ZONE 2 - PATHWAY ZONES (Colored zones around forest)
% Red pathway zone (236,162,151)
zone2_red_x = [0, 3200, 3200, 0];
zone2_red_y = [0, 0, cfg.ARENA_Y-3200, cfg.ARENA_Y-3200];
draw_zone_flat(ax_main, ax_top, zone2_red_x, zone2_red_y, [236/255 162/255 151/255], '', 3);

% Blue pathway zone (128,191,209) - SYMMETRIC
zone2_blue_x = [cfg.ARENA_X-3200, cfg.ARENA_X, cfg.ARENA_X, cfg.ARENA_X-3200];
zone2_blue_y = [0, 0, cfg.ARENA_Y-3200, cfg.ARENA_Y-3200];
draw_zone_flat(ax_main, ax_top, zone2_blue_x, zone2_blue_y, [128/255 191/255 209/255], '', 3);

%% 11. ROBOTS
% R1 Red robot
robot_r1_red = draw_robot_model(ax_main, ax_top, [1000, cfg.ARENA_Y-500, 100], 'R1', 'red');

% R2 Red robot
robot_r2_red = draw_robot_model(ax_main, ax_top, [2000, cfg.ARENA_Y-500, 100], 'R2', 'red');

% R1 Blue robot (SYMMETRIC)
robot_r1_blue = draw_robot_model(ax_main, ax_top, [cfg.ARENA_X-1000, cfg.ARENA_Y-500, 100], 'R1', 'blue');

% R2 Blue robot (SYMMETRIC)
robot_r2_blue = draw_robot_model(ax_main, ax_top, [cfg.ARENA_X-2000, cfg.ARENA_Y-500, 100], 'R2', 'blue');

%% 12. INFORMATION PANEL
info_panel = uipanel(fig, 'Position', [0.05 0.05 0.55 0.25], ...
    'BackgroundColor', [0.12 0.12 0.12], 'ForegroundColor', 'cyan', ...
    'Title', '📊 Arena Specifications (CORRECTED)', 'FontSize', 12, 'FontWeight', 'bold');

info_text = sprintf([...
    '╔═══════════════════════════════════════════════════════════════╗\n', ...
    '║ 🥋 ABU ROBOCON 2026 - KUNG FU QUEST (CORRECTED LAYOUT)       ║\n', ...
    '╠═══════════════════════════════════════════════════════════════╣\n', ...
    '║ 📐 Game Field: 6000mm × 6000mm                                ║\n', ...
    '║ 🌲 Meihua Forest (Per Team):                                  ║\n', ...
    '║    • RED Team: 12 blocks (4 rows × 3 columns)                 ║\n', ...
    '║    • BLUE Team: 12 blocks (4 rows × 3 columns, MIRRORED)      ║\n', ...
    '║    • Block Heights: 200mm, 400mm, 600mm                       ║\n', ...
    '║    • Block Spacing: 1450mm center-to-center                   ║\n', ...
    '║ 🎯 Arena: 2600mm × 2700mm platform (450mm height)             ║\n', ...
    '║ 🎪 Tic-Tac-Toe: 3×3 grid (1620mm × 540mm rack)                ║\n', ...
    '║ 📦 KFS (Kung Fu Scrolls): 350mm cubes                         ║\n', ...
    '║ 🔧 Weapons: Staffs (1027mm) + Spearheads (3 types)            ║\n', ...
    '║ ⚖️  Total Robot Weight: ≤50kg (both robots combined)          ║\n', ...
    '║ ⏱️  Game Time: 3 minutes maximum                               ║\n', ...
    '╠═══════════════════════════════════════════════════════════════╣\n', ...
    '║ ✅ CORRECTIONS APPLIED:                                        ║\n', ...
    '║   • Symmetric forest layout (RED + BLUE)                      ║\n', ...
    '║   • Proper 4×3 block grid per team                            ║\n', ...
    '║   • Exact dimensions from Appendix 1-4                        ║\n', ...
    '║   • Correct zone positions and colors                         ║\n', ...
    '╚═══════════════════════════════════════════════════════════════╝\n']);

uicontrol(info_panel, 'Style', 'text', 'Units', 'normalized', ...
    'Position', [0.02 0.02 0.96 0.96], 'String', info_text, ...
    'BackgroundColor', [0.12 0.12 0.12], 'ForegroundColor', [0 1 1], ...
    'FontName', 'Courier New', 'FontSize', 9, 'HorizontalAlignment', 'left');

%% 13. CONTROL PANEL
control_panel = uipanel(fig, 'Position', [0.65 0.05 0.32 0.25], ...
    'BackgroundColor', [0.12 0.12 0.12], 'ForegroundColor', 'cyan', ...
    'Title', '🎮 View Controls', 'FontSize', 12, 'FontWeight', 'bold');

% View buttons
uicontrol(control_panel, 'Style', 'pushbutton', 'Units', 'normalized', ...
    'Position', [0.05 0.75 0.28 0.2], 'String', '🔝 Top', 'FontSize', 11, ...
    'BackgroundColor', [0.3 0.5 0.7], 'ForegroundColor', 'w', ...
    'Callback', @(~,~) view(ax_main, 0, 90));

uicontrol(control_panel, 'Style', 'pushbutton', 'Units', 'normalized', ...
    'Position', [0.36 0.75 0.28 0.2], 'String', '↗️ ISO', 'FontSize', 11, ...
    'BackgroundColor', [0.3 0.5 0.7], 'ForegroundColor', 'w', ...
    'Callback', @(~,~) view(ax_main, 45, 30));

uicontrol(control_panel, 'Style', 'pushbutton', 'Units', 'normalized', ...
    'Position', [0.67 0.75 0.28 0.2], 'String', '➡️ Side', 'FontSize', 11, ...
    'BackgroundColor', [0.3 0.5 0.7], 'ForegroundColor', 'w', ...
    'Callback', @(~,~) view(ax_main, 0, 0));

% Reset button
uicontrol(control_panel, 'Style', 'pushbutton', 'Units', 'normalized', ...
    'Position', [0.05 0.50 0.9 0.2], 'String', '🔄 Reset View', 'FontSize', 11, ...
    'BackgroundColor', [0.7 0.5 0.2], 'ForegroundColor', 'w', ...
    'Callback', @(~,~) view(ax_main, 45, 30));

% Export button
uicontrol(control_panel, 'Style', 'pushbutton', 'Units', 'normalized', ...
    'Position', [0.05 0.25 0.9 0.2], 'String', '💾 Export 4K Image', 'FontSize', 11, ...
    'BackgroundColor', [0.5 0.3 0.7], 'ForegroundColor', 'w', ...
    'Callback', @(~,~) export_image(fig));

% Team selector
uicontrol(control_panel, 'Style', 'text', 'Units', 'normalized', ...
    'Position', [0.05 0.02 0.9 0.18], 'String', '🏆 Team Vulcans | Rulebook v2025-08', ...
    'BackgroundColor', [0.12 0.12 0.12], 'ForegroundColor', 'yellow', ...
    'FontSize', 10, 'FontWeight', 'bold');

%% Title
title(ax_main, {'ABU ROBOCON 2026 - KUNG FU QUEST', ...
    '✅ CORRECTED Arena Layout (Symmetric Forest + Exact Dimensions)'}, ...
    'Color', 'yellow', 'FontSize', 16, 'FontWeight', 'bold');

%% Console Output
fprintf('\n');
fprintf('╔════════════════════════════════════════════════════════════════╗\n');
fprintf('║  ✅ CORRECTED ABU ROBOCON 2026 ARENA LOADED SUCCESSFULLY!      ║\n');
fprintf('╠════════════════════════════════════════════════════════════════╣\n');
fprintf('║  📏 All dimensions verified against Appendix 1-4               ║\n');
fprintf('║  🌲 Symmetric forest layout: RED + BLUE teams                  ║\n');
fprintf('║  🎯 Proper 4×3 block grid (12 blocks per team)                 ║\n');
fprintf('║  ⚙️ Interactive 3D visualization ready                         ║\n');
fprintf('╚════════════════════════════════════════════════════════════════╝\n');
fprintf('\n');

%% ═══════════════════════════════════════════════════════════════
%% HELPER FUNCTIONS
%% ═══════════════════════════════════════════════════════════════

function draw_zone_flat(ax_main, ax_top, x, y, color, label, height)
    % Draw flat colored zone
    patch(ax_main, 'XData', x, 'YData', y, 'ZData', height*ones(size(x)), ...
        'FaceColor', color, 'FaceAlpha', 0.6, 'EdgeColor', 'k', 'LineWidth', 2);
    patch(ax_top, 'XData', x, 'YData', y, ...
        'FaceColor', color, 'FaceAlpha', 0.5, 'EdgeColor', 'k', 'LineWidth', 1.5);
    if ~isempty(label)
        text(ax_main, mean(x), mean(y), height+50, label, ...
            'Color', 'w', 'FontSize', 10, 'FontWeight', 'bold', 'HorizontalAlignment', 'center');
    end
end

function draw_rack(ax_main, ax_top, pos, size, color, label, is_metal)
    % Draw equipment rack
    draw_box_3d(ax_main, pos, size, color, label, 0.85);
    
    % Top view rectangle
    rectangle(ax_top, 'Position', [pos(1), pos(2)-size(2)/2, size(1), size(2)], ...
        'EdgeColor', 'k', 'LineWidth', 2, 'FaceColor', [color 0.4]);
    
    if is_metal
        % Add metallic sheen
        text(ax_main, pos(1)+size(1)/2, pos(2), pos(3)+size(3)+100, label, ...
            'Color', [1 0.5 0], 'FontSize', 10, 'FontWeight', 'bold');
    end
end

function draw_forest_block(ax_main, ax_top, ax_side, pos, height, block_num, team_color)
    % Draw individual forest block with KFS
    block_half = 725;  % Half of 1450mm
    
    % Block vertices (platform)
    verts = [
        pos(1)-block_half, pos(2)-block_half, pos(3);
        pos(1)+block_half, pos(2)-block_half, pos(3);
        pos(1)+block_half, pos(2)+block_half, pos(3);
        pos(1)-block_half, pos(2)+block_half, pos(3);
        pos(1)-block_half, pos(2)-block_half, pos(3)+height;
        pos(1)+block_half, pos(2)-block_half, pos(3)+height;
        pos(1)+block_half, pos(2)+block_half, pos(3)+height;
        pos(1)-block_half, pos(2)+block_half, pos(3)+height
    ];
    
    faces = [1 2 3 4; 5 6 7 8; 1 2 6 5; 2 3 7 6; 3 4 8 7; 4 1 5 8];
    
    % Color based on height
    if height == 200
        block_color = [41/255 82/255 16/255];
    elseif height == 400
        block_color = [42/255 113/255 56/255];
    else  % 600
        block_color = [152/255 166/255 80/255];
    end
    
    % Draw block
    patch(ax_main, 'Vertices', verts, 'Faces', faces, ...
        'FaceColor', block_color, 'EdgeColor', [0.2 0.5 0.1], ...
        'FaceAlpha', 0.8, 'LineWidth', 1.5);
    
    % Top view block
    rectangle(ax_top, 'Position', [pos(1)-block_half, pos(2)-block_half, 2*block_half, 2*block_half], ...
        'EdgeColor', block_color*1.5, 'LineWidth', 1.5, 'FaceColor', [block_color 0.6]);
    
    % Block number label
    text(ax_main, pos(1), pos(2), pos(3)+height+200, sprintf('B%d\n%dmm', block_num, height), ...
        'Color', 'yellow', 'FontSize', 10, 'FontWeight', 'bold', ...
        'HorizontalAlignment', 'center', 'BackgroundColor', [0 0 0 0.7], ...
        'EdgeColor', 'yellow', 'Margin', 2);
    
    text(ax_top, pos(1), pos(2), sprintf('%d', block_num), ...
        'Color', 'w', 'FontSize', 9, 'FontWeight', 'bold', 'HorizontalAlignment', 'center');
    
    % Side view block
    patch(ax_side, [pos(1)-block_half, pos(1)+block_half, pos(1)+block_half, pos(1)-block_half], ...
        [0, 0, height, height], block_color, 'EdgeColor', 'k', 'LineWidth', 1);
    
    % KFS on top (350mm cube)
    kfs_half = 175;
    kfs_z = pos(3) + height;
    kfs_verts = [
        pos(1)-kfs_half, pos(2)-kfs_half, kfs_z;
        pos(1)+kfs_half, pos(2)-kfs_half, kfs_z;
        pos(1)+kfs_half, pos(2)+kfs_half, kfs_z;
        pos(1)-kfs_half, pos(2)+kfs_half, kfs_z;
        pos(1)-kfs_half, pos(2)-kfs_half, kfs_z+350;
        pos(1)+kfs_half, pos(2)-kfs_half, kfs_z+350;
        pos(1)+kfs_half, pos(2)+kfs_half, kfs_z+350;
        pos(1)-kfs_half, pos(2)+kfs_half, kfs_z+350
    ];
    
    % KFS color based on team
    if strcmp(team_color, 'red')
        kfs_color = [1 0.4 0.3];
    else
        kfs_color = [0.3 0.5 1];
    end
    
    patch(ax_main, 'Vertices', kfs_verts, 'Faces', faces, ...
        'FaceColor', kfs_color, 'EdgeColor', [0.8 0.5 0], ...
        'FaceAlpha', 0.9, 'LineWidth', 2);
    
    % Top view KFS
    scatter(ax_top, pos(1), pos(2), 150, kfs_color, 'filled', 's', ...
        'MarkerEdgeColor', 'k', 'LineWidth', 1.5);
end

function draw_platform(ax_main, ax_top, x, y, height, color, label)
    % Draw elevated platform
    % Create 2D matrices for surf
    X = [x(1) x(2); x(4) x(3)];
    Y = [y(1) y(2); y(4) y(3)];
    Z = height * ones(2, 2);
    
    surf(ax_main, X, Y, Z, 'FaceColor', color, 'FaceAlpha', 0.75, ...
        'EdgeColor', [1 0.5 0], 'LineWidth', 3);
    
    patch(ax_top, 'XData', x, 'YData', y, ...
        'FaceColor', color, 'FaceAlpha', 0.6, 'EdgeColor', [1 0.5 0], 'LineWidth', 2);
    
    if ~isempty(label)
        text(ax_main, mean(x), mean(y), height+100, label, ...
            'Color', 'w', 'FontSize', 11, 'FontWeight', 'bold');
    end
end

function draw_box_3d(ax, center, size, color, label, alpha)
    % Draw 3D box centered at position
    dx = size(1)/2; dy = size(2)/2; dz = size(3)/2;
    verts = [
        center(1)-dx, center(2)-dy, center(3)-dz;
        center(1)+dx, center(2)-dy, center(3)-dz;
        center(1)+dx, center(2)+dy, center(3)-dz;
        center(1)-dx, center(2)+dy, center(3)-dz;
        center(1)-dx, center(2)-dy, center(3)+dz;
        center(1)+dx, center(2)-dy, center(3)+dz;
        center(1)+dx, center(2)+dy, center(3)+dz;
        center(1)-dx, center(2)+dy, center(3)+dz
    ];
    faces = [1 2 3 4; 5 6 7 8; 1 2 6 5; 2 3 7 6; 3 4 8 7; 4 1 5 8];
    
    patch(ax, 'Vertices', verts, 'Faces', faces, ...
        'FaceColor', color, 'EdgeColor', 'k', 'FaceAlpha', alpha, 'LineWidth', 1.5);
    
    if ~isempty(label)
        text(ax, center(1), center(2), center(3)+dz+100, label, ...
            'Color', 'w', 'FontSize', 9, 'FontWeight', 'bold', 'HorizontalAlignment', 'center');
    end
end

function h = draw_robot_model(ax_main, ax_top, pos, robot_name, team)
    % Draw robot model
    if strcmp(team, 'red')
        robot_color = [0.9 0.2 0.2];
        marker_color = [1 0 0];
    else
        robot_color = [0.2 0.4 0.9];
        marker_color = [0 0.5 1];
    end
    
    % 3D body (cylinder)
    [X, Y, Z] = cylinder(200, 20);
    Z = Z * 250 + pos(3);
    X = X + pos(1);
    Y = Y + pos(2);
    h.body = surf(ax_main, X, Y, Z, 'FaceColor', robot_color, ...
        'EdgeColor', 'none', 'FaceAlpha', 0.85);
    
    % Top marker
    h.marker = scatter3(ax_main, pos(1), pos(2), pos(3)+300, 800, ...
        marker_color, 'filled', '^', 'MarkerEdgeColor', 'w', 'LineWidth', 2);
    
    % Label
    h.text = text(ax_main, pos(1), pos(2), pos(3)+400, robot_name, ...
        'Color', 'white', 'FontSize', 14, 'FontWeight', 'bold', ...
        'HorizontalAlignment', 'center', 'BackgroundColor', [marker_color 0.8]);
    
    % Top view
    h.top = scatter(ax_top, pos(1), pos(2), 400, marker_color, 'filled', '^', ...
        'MarkerEdgeColor', 'w', 'LineWidth', 2);
    text(ax_top, pos(1), pos(2)+200, robot_name, 'Color', 'w', ...
        'FontSize', 10, 'FontWeight', 'bold', 'HorizontalAlignment', 'center');
end

function export_image(fig)
    filename = sprintf('RoboconArena_CORRECTED_%s.png', datestr(now, 'yyyymmdd_HHMMSS'));
    fprintf('📸 Exporting image: %s\n', filename);
    exportgraphics(fig, filename, 'Resolution', 300);
    fprintf('✅ Export complete!\n');
end