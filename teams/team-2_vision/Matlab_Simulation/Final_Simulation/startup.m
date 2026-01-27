% startup.m - Run this file to start the Robocon 2026 Simulation
% Place this file in: D:\Robotics Club\Robocon2026\...\files\
%
% Usage: Just run 'startup' in MATLAB Command Window

clear; clc;
fprintf('=== Robocon 2026 Simulation Startup ===\n\n');

% Get the directory where this script is located
scriptDir = fileparts(mfilename('fullpath'));

% Add all source folders to MATLAB path
fprintf('[1/3] Adding source folders to MATLAB path...\n');
addpath(genpath(fullfile(scriptDir, 'src')));
fprintf('      ✓ All source folders added\n\n');

% Verify critical files exist
fprintf('[2/3] Verifying installation...\n');
criticalFiles = {
    fullfile(scriptDir, 'src', 'robocon_arena_sim.m'),
    fullfile(scriptDir, 'src', 'config', 'SimConfig.m'),
    fullfile(scriptDir, 'src', 'core', 'environment', 'Arena.m'),
    fullfile(scriptDir, 'src', 'core', 'robot', 'Robot.m'),
    fullfile(scriptDir, 'src', 'core', 'vision', 'VisionSystem.m'),
    fullfile(scriptDir, 'src', 'core', 'navigation', 'Navigator.m'),
    fullfile(scriptDir, 'src', 'ui', 'SimulationUI.m')
};

allFilesExist = true;
for i = 1:length(criticalFiles)
    if ~exist(criticalFiles{i}, 'file')
        fprintf('      ✗ Missing: %s\n', criticalFiles{i});
        allFilesExist = false;
    end
end

if allFilesExist
    fprintf('      ✓ All critical files found\n\n');
else
    error('Some critical files are missing. Please check your installation.');
end

% Check dataset folder
fprintf('[3/3] Checking dataset folder...\n');
datasetPath = fullfile(scriptDir, 'data', 'dataset');
if exist(datasetPath, 'dir')
    fprintf('      ✓ Dataset folder exists: %s\n', datasetPath);
    
    % Check subfolders
    r1Folder = fullfile(datasetPath, 'R1');
    realFolder = fullfile(datasetPath, 'real');
    fakeFolder = fullfile(datasetPath, 'fake');
    
    if exist(r1Folder, 'dir')
        r1Files = [dir(fullfile(r1Folder, '*.png')); dir(fullfile(r1Folder, '*.jpg'))];
        fprintf('      ✓ R1 folder: %d images found\n', length(r1Files));
    else
        fprintf('      ⚠ R1 folder not found (will use placeholders)\n');
    end
    
    if exist(realFolder, 'dir')
        realFiles = [dir(fullfile(realFolder, '*.png')); dir(fullfile(realFolder, '*.jpg'))];
        fprintf('      ✓ Real folder: %d images found\n', length(realFiles));
    else
        fprintf('      ⚠ Real folder not found (will use placeholders)\n');
    end
    
    if exist(fakeFolder, 'dir')
        fakeFiles = [dir(fullfile(fakeFolder, '*.png')); dir(fullfile(fakeFolder, '*.jpg'))];
        fprintf('      ✓ Fake folder: %d images found\n', length(fakeFiles));
    else
        fprintf('      ⚠ Fake folder not found (will use placeholders)\n');
    end
else
    fprintf('      ⚠ Dataset folder not found: %s\n', datasetPath);
    fprintf('      Simulation will use placeholder images\n');
end

fprintf('\n=== Setup Complete ===\n\n');
fprintf('Starting simulation...\n');
fprintf('Press Ctrl+C to stop if needed\n\n');

% Run the simulation
try
    robocon_arena_sim();
catch ME
    fprintf('\n=== Simulation Error ===\n');
    fprintf('Error: %s\n', ME.message);
    fprintf('File: %s (line %d)\n', ME.stack(1).file, ME.stack(1).line);
    fprintf('\nTo fix:\n');
    fprintf('1. Check that dataset path in SimConfig.m is correct\n');
    fprintf('2. Verify all .m files are in correct folders\n');
    fprintf('3. See docs/README.md for troubleshooting\n');
end
