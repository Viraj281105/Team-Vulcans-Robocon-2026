clc;
clear;
close all;

%% ============================================================
%  ROBOCON 2026 – ARENA STL LOADER & VISUALIZATION PIPELINE
%  - Robust STL import
%  - Auto-orientation normalization
%  - Centering & scaling
%  - Stable rendering (no disappearing faces)
%  - Free 3D navigation
%  - Diagnostics for future robotics integration
% ============================================================

%% ------------------ STL FILE PATH ----------------------------
stlFile = "D:\Robotics Club\Robocon2026\Team-Vulcans-Robocon-2026\teams\team-2_vision\Matlab_Simulation\Arena Simulation\AreanaLayout_2026_v1.STL";
assert(isfile(stlFile), "❌ STL file not found. Check path.");

fprintf("✅ STL found. Loading...\n");

%% ------------------ LOAD STL -------------------------------
arena = stlread(stlFile);

% Extract Faces & Vertices safely across MATLAB versions
if isstruct(arena) || isa(arena,'triangulation')
    F = arena.ConnectivityList;
    V = arena.Points;
else
    error("❌ Unknown STL format returned by stlread.");
end

fprintf("✔ Vertices: %d | Faces: %d\n", size(V,1), size(F,1));

%% ------------------ DIAGNOSTICS ------------------------------
minV = min(V);
maxV = max(V);
span = maxV - minV;

fprintf("\n--- Geometry Diagnostics ---\n");
fprintf("X range: %.1f mm\n", span(1));
fprintf("Y range: %.1f mm\n", span(2));
fprintf("Z range: %.1f mm\n", span(3));

% Identify likely vertical axis (smallest span)
[~, verticalAxis] = min(span);
axesLabel = ["X","Y","Z"];
fprintf("Likely vertical axis: %s\n", axesLabel(verticalAxis));

%% ------------------ AUTO ORIENTATION FIX ---------------------
% Goal: Ground plane → XY , Height → Z

Vh = [V ones(size(V,1),1)];

switch verticalAxis
    case 1   % X is vertical → rotate about Y
        fprintf("🔄 Rotating about Y axis...\n");
        T = makehgtform('yrotate', deg2rad(90));

    case 2   % Y is vertical → rotate about X
        fprintf("🔄 Rotating about X axis...\n");
        T = makehgtform('xrotate', deg2rad(-90));

    case 3   % Z already vertical → no rotation needed
        fprintf("✅ Z axis already vertical. No rotation needed.\n");
        T = eye(4);
end

V_rot = (T * Vh')';
V = V_rot(:,1:3);


%% ------------------ AXIS MIRROR FIX --------------------------
% CAD handedness correction (flip X axis)

V(:,1) = -V(:,1);
fprintf("🔄 X-axis mirrored to correct handedness.\n");

%% ------------------ CENTER MODEL AT ORIGIN -------------------
center = mean(V,1);
V = V - center;

fprintf("✔ Arena centered at origin.\n");

%% ------------------ VISUALIZATION ----------------------------
figure('Color','w','Name','Robocon Arena Viewer');

h = patch('Faces', F, 'Vertices', V, ...
      'FaceColor', [0.82 0.82 0.82], ...
      'EdgeColor', 'none', ...
      'FaceLighting','gouraud', ...
      'BackFaceLighting','reverselit');   % Critical fix

% Lighting
camlight('headlight');
camlight(45,30);
lighting gouraud;

% Axes control
axis equal;
axis tight;
axis vis3d;
grid on;

xlabel('X (mm)');
ylabel('Y (mm)');
zlabel('Z (mm)');
title('Robocon Arena 2026 – Digital Twin View');

% Camera behavior
view(45,25);
rotate3d on;
camproj perspective;
camzoom(0.9);

%% ------------------ FLOOR REFERENCE GRID ---------------------
hold on;
floorZ = min(V(:,3));
[Xg,Yg] = meshgrid(linspace(min(V(:,1)),max(V(:,1)),10), ...
                   linspace(min(V(:,2)),max(V(:,2)),10));
Zg = floorZ * ones(size(Xg));
surf(Xg,Yg,Zg, ...
    'FaceAlpha',0.05, ...
    'EdgeColor',[0.5 0.5 0.5]);

%% ------------------ FINAL STATUS -----------------------------
fprintf("\n🚀 Arena loaded successfully.\n");
fprintf("🖱️ Mouse Controls:\n");
fprintf("   - Rotate: Left drag\n");
fprintf("   - Pan: Middle drag\n");
fprintf("   - Zoom: Scroll wheel\n");
fprintf("=============================================\n");
