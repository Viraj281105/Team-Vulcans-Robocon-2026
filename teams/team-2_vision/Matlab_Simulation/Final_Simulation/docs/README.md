# Robocon 2026 Simulation - Refactored Code Structure

## Overview
This is a modular refactoring of the Robocon 2026 3D simulation with time-aware hybrid A* navigation. The code has been split into logical, manageable components for better maintainability and extensibility.

## File Structure

### Main Entry Point
- **robocon_arena_sim.m** - Main simulation entry point that initializes and coordinates all components

### Configuration
- **SimConfig.m** - Central configuration class containing all constants, parameters, and settings

### Core Simulation Components

#### Arena & Environment
- **Arena.m** - Manages the forest blocks, KFS placement, and ground truth
- **ImageLoader.m** - Utility for loading oracle bone character images

#### Robot
- **Robot.m** - Robot state management, movement, and positioning

#### Vision System
- **VisionSystem.m** - HSV+ML vision pipeline simulation with detection tracking
- **OcclusionChecker.m** - Ray-box intersection tests for occlusion detection

#### Navigation System
- **Navigator.m** - High-level autonomous navigation controller
- **NavigationPlanner.m** - Target selection algorithms (R2 selection, exploration, exit planning)
- **PathPlanner.m** - A* pathfinding with terrain-aware costs
- **TerrainCost.m** - Terrain cost calculation utilities

#### User Interface
- **SimulationUI.m** - UI management, rendering coordination, and camera control
- **Renderer.m** - 3D rendering utilities for drawing scene elements
- **InputHandler.m** - Keyboard input processing and command handling

## Architecture

```
robocon_arena_sim (Main)
    ├── SimConfig (Configuration)
    ├── Arena (Environment)
    │   └── ImageLoader (Images)
    ├── Robot (State & Movement)
    ├── VisionSystem (Perception)
    │   └── OcclusionChecker (Visibility)
    ├── Navigator (Autonomous Control)
    │   ├── NavigationPlanner (Target Selection)
    │   ├── PathPlanner (A* Search)
    │   └── TerrainCost (Cost Calculation)
    └── SimulationUI (Interface)
        ├── Renderer (Graphics)
        └── InputHandler (Input)
```

## Class Responsibilities

### SimConfig
- Stores all simulation constants and parameters
- Provides configuration queries (paths, penalties, etc.)
- Single source of truth for tunable values

### Arena
- Manages the 4x3 forest grid with varying heights
- Handles KFS (oracle bone scroll) placement and randomization
- Maintains ground truth state
- Provides spatial queries (boundaries, block info, height map)

### Robot
- Tracks 3D position and orientation
- Handles movement with collision detection
- Manages grid position and block climbing
- Records path history and statistics

### VisionSystem
- Simulates camera-based detection with HSV filtering
- Implements ML confidence scoring
- Tracks detected KFS with time-based confirmation
- Updates grid belief state from camera view
- Maintains detection statistics

### Navigator
- High-level autonomous navigation state machine
- Manages COLLECT/EXIT/WAIT modes
- Updates belief map from vision
- Handles stuck detection and replanning
- Coordinates with PathPlanner for movement

### NavigationPlanner
- Implements target selection algorithms
- Hybrid cost function for R2 selection (distance + exit proximity)
- Exploration strategy for unknown cells
- Exit route planning with R1 avoidance

### PathPlanner
- A* search implementation with terrain costs
- Mode-specific obstacle handling (COLLECT vs EXIT)
- R1 waiting penalties and R2 exit penalties
- Cell traversability evaluation

### TerrainCost
- Height-based movement cost calculations
- Uphill/downhill cost asymmetry (T_UP, T_DOWN)
- Path cost accumulation

### SimulationUI
- Manages MATLAB figure and axes
- Coordinates rendering of overview and camera views
- Auto-tilt camera feature
- Help overlay and HUD display

### Renderer
- Static utility methods for 3D drawing
- Block rendering with height-based colors
- KFS cubes with oracle bone textures
- Robot visualization with direction arrow
- Camera view culling and visibility

### InputHandler
- Keyboard event processing
- Manual robot control (W/A/S/D/Q/E)
- Camera control (I/K/Z/X)
- Mode toggles (V/N/P/H)
- Export functionality (L)

## Key Features

1. **Modular Design**: Each component has a single, well-defined responsibility
2. **Configuration Centralization**: All tunable parameters in SimConfig
3. **Object-Oriented**: Proper use of classes with clear interfaces
4. **Separation of Concerns**: Rendering, simulation logic, and input handling are separated
5. **Maintainability**: Easy to modify individual components without affecting others

## Usage

```matlab
% Simply run the main function
robocon_arena_sim()
```

## Controls

### Manual Robot Control
- W/S: Move Forward/Backward
- A/D: Strafe Left/Right
- Q/E: Rotate Left/Right
- SPACE: Climb to current block
- C: Descend to ground

### Camera Control
- I/K: Pan camera (yaw)
- Z/X: Pitch camera up/down
- V: Toggle auto-tilt (centers on KFS)

### Simulation Control
- N: Toggle autonomous navigation
- R: Reset robot
- T: Randomize KFS placement
- P: Toggle path recording
- L: Export trajectory and mission data
- H: Toggle help overlay
- ESC: Quit simulation

## Extending the Code

### Adding New Features

**New KFS Type:**
1. Update `SimConfig.TOTAL_*` constants
2. Modify `Arena.randomizeKFS()` placement logic
3. Update detection logic in `VisionSystem`
4. Adjust navigation costs in `PathPlanner.evaluateCell()`

**New Navigation Mode:**
1. Add mode to `Navigator.mode` property
2. Implement mode logic in `Navigator.planAndMove()`
3. Add target selection in `NavigationPlanner`
4. Update A* costs in `PathPlanner.evaluateCell()`

**New Camera Feature:**
1. Add property to `SimulationUI`
2. Implement feature in camera calculation methods
3. Add keyboard shortcut in `InputHandler.handleKey()`
4. Update help text

### Tuning Parameters

All tunable parameters are in `SimConfig.m`:
- Movement speeds: `MOVE_SPEED`, `ROTATION_SPEED`
- Vision: `DETECTION_RANGE`, `DETECTION_FOV`, `CONF_THRESHOLD`
- Navigation weights: `ALPHA0/1`, `BETA0/1`
- Terrain costs: `T_UP`, `T_DOWN`, `T_PICK`
- Time limits: `TIME_LIMIT`, `TIME_BUFFER`

## Testing

Each component can be tested independently:

```matlab
% Test configuration
cfg = SimConfig();
assert(cfg.ARENA_X == 6000);

% Test arena generation
arena = Arena(cfg);
assert(numel(arena.blocks) == 12);

% Test vision occlusion
isOcc = OcclusionChecker.isOccluded(p1, p2, blocks, blockSize, targetId);

% Test terrain costs
cost = TerrainCost.estimate([1,1], [4,3], heightMap, cfg);
```

## Performance Considerations

- **A* Search**: Limited to `ROWS*COLS*4` iterations to prevent infinite loops
- **Rendering**: Camera view culling based on `CAMERA_RANGE`
- **Detection**: Only processes KFS within `DETECTION_RANGE` and FOV
- **Path History**: Only recorded when `recordingPath` is enabled

## Future Improvements

1. **Multi-robot support**: Extend Robot and Navigator to handle multiple robots
2. **Dynamic obstacles**: Add moving obstacles to Arena
3. **Advanced ML vision**: Replace simulated confidence with real ML model
4. **Communication**: Add robot-to-robot communication simulation
5. **3D path planning**: Extend PathPlanner to plan in 3D space
6. **Performance profiling**: Add timing metrics for each component

## Dependencies

- MATLAB R2019b or later
- Image Processing Toolbox (for image loading)
- No external packages required

## License

[Add your license information here]
