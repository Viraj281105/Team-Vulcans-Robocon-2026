classdef SimConfig < handle
    % ===================================================================
    % SIMCONFIG
    % -------------------------------------------------------------------
    % Central configuration authority for the ROBOCON 2026 simulator.
    %
    % Responsibilities:
    %   - Defines all physical dimensions (arena, blocks, robot).
    %   - Controls camera optics and perception ranges.
    %   - Governs navigation heuristics and cost shaping.
    %   - Stores dataset paths and asset lookup helpers.
    %   - Encapsulates tunable behavior knobs.
    %
    % Architectural role:
    %   - Single source of truth for the entire simulation stack.
    %   - Eliminates magic numbers from runtime logic.
    %   - Enables rapid experimentation and reproducibility.
    % ===================================================================
    
    properties (Constant)
        
        % ===================================================================
        % Dataset Root Path
        % -------------------------------------------------------------------
        % Dynamically resolves dataset path relative to this file location.
        % Allows project portability across machines.
        % ===================================================================
        DATASET_PATH = fullfile( ...
            fileparts(fileparts(fileparts(mfilename('fullpath')))), ...
            'data', 'dataset')
        
        % ===================================================================
        % Arena Dimensions (millimeters)
        % -------------------------------------------------------------------
        % Physical size of competition arena.
        % ===================================================================
        ARENA_X = 6000
        ARENA_Y = 7300
        ARENA_Z = 2000
        
        % ===================================================================
        % Forest / Block Geometry
        % -------------------------------------------------------------------
        % Grid configuration and physical dimensions of forest blocks.
        % ===================================================================
        BLOCK_SIZE  = 1200
        FOREST_COLS = 3
        FOREST_ROWS = 4
        KFS_SIZE    = 350        % Oracle bone cube size (mm)
        
        % ===================================================================
        % Robot Physical Dimensions (millimeters)
        % -------------------------------------------------------------------
        R2_WIDTH  = 800
        R2_LENGTH = 800
        R2_HEIGHT = 800
        
        % ===================================================================
        % Robot Motion Parameters
        % -------------------------------------------------------------------
        % Linear movement speed and angular rotation step.
        % ===================================================================
        MOVE_SPEED     = 150
        ROTATION_SPEED = 5
        
        % ===================================================================
        % Camera Parameters
        % -------------------------------------------------------------------
        % Defines robot-mounted camera geometry and control sensitivity.
        % ===================================================================
        CAMERA_HEIGHT_OFFSET = 800
        CAMERA_FOV_H         = 100
        CAMERA_FOV_V         = 65
        CAMERA_RANGE         = 3000
        PITCH_STEP           = 5
        YAW_STEP             = 5
        
        % ===================================================================
        % Vision System Parameters
        % -------------------------------------------------------------------
        % Controls detection limits and temporal stability.
        % ===================================================================
        CONF_THRESHOLD      = 0.50
        DETECTION_RANGE     = 2500
        DETECTION_FOV       = 90
        MIN_DETECTION_TIME  = 0.5
        
        % ===================================================================
        % Navigation Parameters
        % -------------------------------------------------------------------
        CAPACITY    = 2          % Maximum items robot can carry
        TIME_LIMIT  = 90         % Mission duration (seconds)
        TIME_BUFFER = 15         % Emergency exit buffer
        
        % ===================================================================
        % Terrain Cost Scaling
        % -------------------------------------------------------------------
        % Used by TerrainCost and PathPlanner.
        % ===================================================================
        T_UP   = 1.0             % Cost per unit ascent
        T_DOWN = 2.0             % Cost per unit descent
        T_PICK = 0.5             % Cost to pick R2
        
        % ===================================================================
        % R1 Waiting Model
        % -------------------------------------------------------------------
        % Models uncertainty when R1 blocks exit.
        % ===================================================================
        WAIT_TIME      = 5
        R1_CLEAR_PROB  = 0.8
        
        % ===================================================================
        % Dynamic Navigation Weights
        % -------------------------------------------------------------------
        % Interpolated based on carry load.
        % ===================================================================
        ALPHA0 = 3.0
        BETA0  = 0.5
        ALPHA1 = 0.5
        BETA1  = 3.0
        
        % ===================================================================
        % Vision Confidence Thresholds
        % -------------------------------------------------------------------
        VISION_CONFIDENCE_THRESHOLD = 0.7
        
        % ===================================================================
        % Rendering Colors (RGB normalized)
        % -------------------------------------------------------------------
        COLOR_KFS_RED    = [1 0 0]
        COLOR_KFS_BLUE   = [0 0 1]
        COLOR_KFS_WHITE  = [1 1 1]      % R1 scroll color
        
        COLOR_PATHWAY    = [236 162 151]/255
        COLOR_ROBOT      = [0.4 0.4 0.4]
        
        COLOR_BLOCK_LOW  = [41 82 16]/255
        COLOR_BLOCK_MID  = [42 113 56]/255
        COLOR_BLOCK_HIGH = [152 166 80]/255
        
        % ===================================================================
        % KFS Population Counts
        % -------------------------------------------------------------------
        TOTAL_R2_REAL = 4
        TOTAL_R1      = 3
        TOTAL_FAKE    = 1
    end
    
    properties
        
        % ===================================================================
        % Exit Cells (Grid Coordinates)
        % -------------------------------------------------------------------
        % Defines legal exit locations in forest grid.
        % ===================================================================
        exitCells = [4 1; 4 3]
        
        % ===================================================================
        % Initial Robot Spawn Configuration
        % -------------------------------------------------------------------
        % Special value -1 enables ground start outside forest.
        % ===================================================================
        initialBlockId = -1
        initialRobotX  = 3000
        initialRobotY  = 500
        initialRobotZ  = 0
        
        % ===================================================================
        % Movement Enforcement Flags
        % -------------------------------------------------------------------
        % Allows bypassing constraints for debugging or research.
        % ===================================================================
        ENFORCE_BLOCK_RESTRICTIONS = false
        ENFORCE_R1_WAIT            = false
        
        % ===================================================================
        % Forest Height Map (millimeters)
        % -------------------------------------------------------------------
        % Defines elevation of each block in grid.
        % ===================================================================
        heightMap = [ ...
            400 200 400; ...
            200 400 600; ...
            400 600 400; ...
            200 400 200 ]
    end
    
    methods
        
        % ===================================================================
        % Constructor
        % -------------------------------------------------------------------
        % Verifies dataset availability at startup.
        % ===================================================================
        function obj = SimConfig()
            
            if ~exist(obj.DATASET_PATH, 'dir')
                warning('SimConfig:DatasetNotFound', ...
                    'Dataset folder not found at: %s\nWill use placeholder images.', ...
                    obj.DATASET_PATH);
            else
                fprintf('[CONFIG] Dataset path: %s\n', obj.DATASET_PATH);
            end
        end
        
        
        % ===================================================================
        % Dataset Path Helpers
        % -------------------------------------------------------------------
        % Provides structured access to dataset subfolders.
        % ===================================================================
        function path = getR1Folder(obj)
            path = fullfile(obj.DATASET_PATH, 'R1');
        end
        
        function path = getRealFolder(obj)
            path = fullfile(obj.DATASET_PATH, 'real');
        end
        
        function path = getFakeFolder(obj)
            path = fullfile(obj.DATASET_PATH, 'fake');
        end
        
        
        % ===================================================================
        % Derived Penalty Models
        % -------------------------------------------------------------------
        % Computes expected penalties used in navigation heuristics.
        % ===================================================================
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
