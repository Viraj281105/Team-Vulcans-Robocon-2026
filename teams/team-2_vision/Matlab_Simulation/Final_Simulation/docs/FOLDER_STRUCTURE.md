# Robocon 2026 Simulation - Folder Structure

## Recommended Project Organization

```
Robocon2026/
│
├── README.md                          # Main documentation
├── QUICK_REFERENCE.md                 # Developer quick reference
├── FOLDER_STRUCTURE.md                # This file
│
├── robocon_arena_sim.m                # Main entry point - RUN THIS FILE
│
├── config/                            # Configuration
│   └── SimConfig.m                    # All simulation parameters
│
├── core/                              # Core simulation components
│   ├── environment/                   # Arena and environment
│   │   ├── Arena.m                    # Forest blocks and KFS management
│   │   └── ImageLoader.m              # Oracle bone image loading
│   │
│   ├── robot/                         # Robot control
│   │   └── Robot.m                    # Robot state and movement
│   │
│   ├── vision/                        # Vision system
│   │   ├── VisionSystem.m             # Detection pipeline
│   │   └── OcclusionChecker.m         # Visibility checking
│   │
│   └── navigation/                    # Navigation system
│       ├── Navigator.m                # Main navigation controller
│       ├── NavigationPlanner.m        # Target selection algorithms
│       ├── PathPlanner.m              # A* pathfinding
│       └── TerrainCost.m              # Cost calculations
│
├── ui/                                # User interface
│   ├── SimulationUI.m                 # UI management and rendering
│   ├── Renderer.m                     # 3D rendering utilities
│   └── InputHandler.m                 # Keyboard input handling
│
├── data/                              # Data folder (you need to create this)
│   └── dataset/                       # Oracle bone images
│       ├── R1/                        # R1 scroll images (edges/corners)
│       │   ├── image1.png
│       │   ├── image2.png
│       │   └── ...
│       ├── real/                      # R2 real scroll images
│       │   ├── image1.png
│       │   ├── image2.png
│       │   └── ...
│       └── fake/                      # R2 fake scroll images
│           ├── image1.png
│           └── ...
│
└── output/                            # Generated output (auto-created)
    ├── trajectory_YYYYMMDD_HHMMSS.csv
    └── mission_data_YYYYMMDD_HHMMSS.mat
```

## Option 1: Flat Structure (Simplest)

If you prefer to keep all files in one folder (easiest to set up):

```
Robocon2026/
│
├── README.md
├── QUICK_REFERENCE.md
├── FOLDER_STRUCTURE.md
│
├── robocon_arena_sim.m          # Main entry point
├── SimConfig.m
├── Arena.m
├── Robot.m
├── VisionSystem.m
├── Navigator.m
├── NavigationPlanner.m
├── PathPlanner.m
├── TerrainCost.m
├── SimulationUI.m
├── Renderer.m
├── InputHandler.m
├── ImageLoader.m
├── OcclusionChecker.m
│
└── dataset/                     # Oracle bone images
    ├── R1/
    ├── real/
    └── fake/
```

**To use flat structure:**
1. Put all .m files in the same folder
2. Update `SimConfig.m` line 11 to point to your dataset:
   ```matlab
   DATASET_PATH = './dataset'  % If dataset is in same folder
   ```
3. Run `robocon_arena_sim` in MATLAB

## Option 2: Organized Structure (Recommended)

For better organization as project grows:

```
Robocon2026/
│
├── docs/                        # Documentation
│   ├── README.md
│   ├── QUICK_REFERENCE.md
│   └── FOLDER_STRUCTURE.md
│
├── src/                         # Source code
│   ├── robocon_arena_sim.m     # Main entry point
│   │
│   ├── config/
│   │   └── SimConfig.m
│   │
│   ├── core/
│   │   ├── environment/
│   │   │   ├── Arena.m
│   │   │   └── ImageLoader.m
│   │   ├── robot/
│   │   │   └── Robot.m
│   │   ├── vision/
│   │   │   ├── VisionSystem.m
│   │   │   └── OcclusionChecker.m
│   │   └── navigation/
│   │       ├── Navigator.m
│   │       ├── NavigationPlanner.m
│   │       ├── PathPlanner.m
│   │       └── TerrainCost.m
│   │
│   └── ui/
│       ├── SimulationUI.m
│       ├── Renderer.m
│       └── InputHandler.m
│
├── data/                        # Input data
│   └── dataset/
│       ├── R1/
│       ├── real/
│       └── fake/
│
└── output/                      # Generated output
    ├── trajectories/
    └── mission_data/
```

**To use organized structure:**
1. Create the folder structure
2. Place files in appropriate folders
3. Add all folders to MATLAB path:
   ```matlab
   addpath(genpath('Robocon2026'));
   ```
4. Update `SimConfig.m` line 11:
   ```matlab
   DATASET_PATH = '../data/dataset'  % Relative to src/config/
   ```
5. Run from MATLAB: `robocon_arena_sim`

## Current Files You Have

Based on the refactored code, you have these 16 files:

### MATLAB Files (.m) - 14 files
1. robocon_arena_sim.m
2. SimConfig.m
3. Arena.m
4. Robot.m
5. VisionSystem.m
6. Navigator.m
7. NavigationPlanner.m
8. PathPlanner.m
9. TerrainCost.m
10. SimulationUI.m
11. Renderer.m
12. InputHandler.m
13. ImageLoader.m
14. OcclusionChecker.m

### Documentation (.md) - 2 files
15. README.md
16. QUICK_REFERENCE.md

## Setup Instructions

### Quick Setup (Flat Structure)

```bash
# 1. Create project folder
mkdir Robocon2026
cd Robocon2026

# 2. Copy all files here
# (Copy the 16 files you received)

# 3. Create dataset folder
mkdir -p dataset/R1 dataset/real dataset/fake

# 4. Copy your oracle bone images to:
#    dataset/R1/     (R1 scroll images)
#    dataset/real/   (R2 real images)
#    dataset/fake/   (R2 fake images)

# 5. Open MATLAB in this folder and run:
#    >> robocon_arena_sim
```

### Detailed Setup (Organized Structure)

```bash
# 1. Create folder structure
mkdir -p Robocon2026/{docs,src/{config,core/{environment,robot,vision,navigation},ui},data/dataset/{R1,real,fake},output}

# 2. Move files to appropriate locations
# Documentation
mv README.md QUICK_REFERENCE.md FOLDER_STRUCTURE.md docs/

# Main entry
mv robocon_arena_sim.m src/

# Config
mv SimConfig.m src/config/

# Environment
mv Arena.m ImageLoader.m src/core/environment/

# Robot
mv Robot.m src/core/robot/

# Vision
mv VisionSystem.m OcclusionChecker.m src/core/vision/

# Navigation
mv Navigator.m NavigationPlanner.m PathPlanner.m TerrainCost.m src/core/navigation/

# UI
mv SimulationUI.m Renderer.m InputHandler.m src/ui/

# 3. Copy oracle bone images to data/dataset/

# 4. In MATLAB:
cd Robocon2026
addpath(genpath('src'));
robocon_arena_sim
```

## Path Configuration

Update the dataset path in `SimConfig.m` based on your structure:

### Flat Structure
```matlab
% Line 11 in SimConfig.m
DATASET_PATH = './dataset'
% or
DATASET_PATH = 'dataset'
```

### Organized Structure (if running from src/)
```matlab
% Line 11 in SimConfig.m
DATASET_PATH = '../data/dataset'
```

### Absolute Path (most reliable)
```matlab
% Line 11 in SimConfig.m (Windows)
DATASET_PATH = 'D:\Robotics Club\Robocon2026\Team-Vulcans-Robocon-2026\teams\team-2_vision\dataset'

% Line 11 in SimConfig.m (Linux/Mac)
DATASET_PATH = '/home/user/Robocon2026/data/dataset'
```

## File Dependencies

```
robocon_arena_sim.m
├── SimConfig.m
├── Arena.m
│   └── ImageLoader.m
├── Robot.m
├── VisionSystem.m
│   └── OcclusionChecker.m
├── Navigator.m
│   ├── NavigationPlanner.m
│   ├── PathPlanner.m
│   │   └── TerrainCost.m
│   └── TerrainCost.m
└── SimulationUI.m
    ├── Renderer.m
    │   └── OcclusionChecker.m
    └── InputHandler.m
```

## Running the Simulation

### Method 1: MATLAB GUI
1. Open MATLAB
2. Navigate to project folder
3. Type `robocon_arena_sim` in Command Window
4. Press Enter

### Method 2: MATLAB Script
```matlab
% setup.m
clear; clc;
addpath(genpath(pwd));  % Add all subfolders
robocon_arena_sim;      % Run simulation
```

### Method 3: Command Line (if using organized structure)
```matlab
% In MATLAB Command Window
cd('path/to/Robocon2026');
addpath(genpath('src'));
robocon_arena_sim;
```

## Troubleshooting

### "Undefined function or variable"
**Problem:** MATLAB can't find the class files  
**Solution:** Add all folders to path:
```matlab
addpath(genpath('path/to/Robocon2026'));
```

### "No such file or directory" (for images)
**Problem:** Dataset path is incorrect  
**Solution:** Update `DATASET_PATH` in `SimConfig.m` to point to your dataset folder

### "Class not found"
**Problem:** .m files not in MATLAB path  
**Solution:** Ensure all .m files are in current directory or added to path

## Best Practices

1. **Use Flat Structure** during development for simplicity
2. **Switch to Organized Structure** when project grows
3. **Keep dataset path configurable** in SimConfig.m
4. **Use version control** (Git) - add `output/` to .gitignore
5. **Document changes** in README.md

## Minimal Working Example

The absolute minimum you need to run:

```
MySimulation/
├── robocon_arena_sim.m
├── SimConfig.m
├── Arena.m
├── ImageLoader.m
├── Robot.m
├── VisionSystem.m
├── OcclusionChecker.m
├── Navigator.m
├── NavigationPlanner.m
├── PathPlanner.m
├── TerrainCost.m
├── SimulationUI.m
├── Renderer.m
├── InputHandler.m
└── dataset/
    ├── R1/
    ├── real/
    └── fake/
```

Just put all 14 .m files and dataset folder together, update the dataset path in SimConfig.m, and run!
