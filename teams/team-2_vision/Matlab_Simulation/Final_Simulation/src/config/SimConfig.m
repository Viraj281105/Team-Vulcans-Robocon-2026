classdef SimConfig < handle
    % SIMCONFIG Configuration class for Robocon 2026 Simulation
    % Contains all constants, paths, and parameters
    
    properties (Constant)
        % Dataset paths - UPDATED FOR ORGANIZED STRUCTURE
        % This path is relative to the 'files' folder
        % From: files/src/config/SimConfig.m
        % To:   files/data/dataset/
        DATASET_PATH = fullfile(fileparts(fileparts(fileparts(mfilename('fullpath')))), 'data', 'dataset')
        
        % Arena dimensions (mm)
        ARENA_X = 6000
        ARENA_Y = 7300
        ARENA_Z = 2000
        
        % Block/Forest configuration
        BLOCK_SIZE = 1200
        FOREST_COLS = 3
        FOREST_ROWS = 4
        KFS_SIZE = 350  % mm
        
        % Robot dimensions (mm)
        R2_WIDTH = 800
        R2_LENGTH = 800
        R2_HEIGHT = 800
        
        % Robot movement
        MOVE_SPEED = 150
        ROTATION_SPEED = 5
        
        % Camera parameters
        CAMERA_HEIGHT_OFFSET = 800
        CAMERA_FOV_H = 100
        CAMERA_FOV_V = 65
        CAMERA_RANGE = 3000
        PITCH_STEP = 5
        YAW_STEP = 5
        
        % Vision parameters
        CONF_THRESHOLD = 0.50
        DETECTION_RANGE = 2500
        DETECTION_FOV = 90  % degrees
        MIN_DETECTION_TIME = 0.5  % seconds
        
        % Navigation parameters
        CAPACITY = 2
        TIME_LIMIT = 90  % seconds
        TIME_BUFFER = 15  % seconds
        
        % Terrain costs
        T_UP = 1.0      % per 20mm height increase
        T_DOWN = 2.0    % per 20mm height decrease
        T_PICK = 0.5    % picking time
        
        % R1 waiting model
        WAIT_TIME = 5
        R1_CLEAR_PROB = 0.8
        
        % Dynamic weights
        ALPHA0 = 3.0   % block weight when empty
        BETA0 = 0.5    % exit weight when empty
        ALPHA1 = 0.5   % block weight when full
        BETA1 = 3.0    % exit weight when full
        
        % Vision confidence
        VISION_CONFIDENCE_THRESHOLD = 0.7
        
        % Colors
        COLOR_KFS_RED = [1 0 0]
        COLOR_KFS_BLUE = [0 0 1]
        COLOR_PATHWAY = [236 162 151]/255
        COLOR_ROBOT = [0.4 0.4 0.4]
        COLOR_BLOCK_LOW = [41 82 16]/255
        COLOR_BLOCK_MID = [42 113 56]/255
        COLOR_BLOCK_HIGH = [152 166 80]/255
        
        % KFS counts
        TOTAL_R2_REAL = 4
        TOTAL_R1 = 3
        TOTAL_FAKE = 1
    end
    
    properties
        % Exit cells (grid coordinates)
        exitCells = [4 1; 4 3]
        
        % Initial block ID
        initialBlockId = 2
        
        % Height map (mm)
        heightMap = [200 200 400; ...
                     200 400 600; ...
                     400 600 400; ...
                     200 400 200]
    end
    
    methods
        function obj = SimConfig()
            % Constructor - verify dataset path exists
            if ~exist(obj.DATASET_PATH, 'dir')
                warning('SimConfig:DatasetNotFound', ...
                    'Dataset folder not found at: %s\nWill use placeholder images.', ...
                    obj.DATASET_PATH);
            else
                fprintf('[CONFIG] Dataset path: %s\n', obj.DATASET_PATH);
            end
        end
        
        function path = getR1Folder(obj)
            path = fullfile(obj.DATASET_PATH, 'R1');
        end
        
        function path = getRealFolder(obj)
            path = fullfile(obj.DATASET_PATH, 'real');
        end
        
        function path = getFakeFolder(obj)
            path = fullfile(obj.DATASET_PATH, 'fake');
        end
        
        function penalty = getR1WaitExpected(obj)
            penalty = obj.WAIT_TIME / obj.R1_CLEAR_PROB;
        end
        
        function penalty = getR2ExitPenalty(obj)
            penalty = 5.0 * obj.getR1WaitExpected();
        end
        
        function penalty = getR1SoftPenalty(obj)
            penalty = 0.2 * obj.getR1WaitExpected();
        end
    end
end
