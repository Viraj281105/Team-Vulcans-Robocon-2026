# Installation Instructions for Your Organized Structure

## Current Folder Structure ✓

You have correctly set up the organized structure:

```
D:\...\files\
├── data\
│   └── dataset\
│       ├── fake\
│       ├── R1\
│       └── real\
├── docs\
│   ├── QUICK_REFERENCE.md
│   └── README.md
├── output\
│   ├── mission_data\
│   └── trajectories\
└── src\
    ├── robocon_arena_sim.m
    ├── config\
    │   └── SimConfig.m
    ├── core\
    │   ├── environment\
    │   │   ├── Arena.m
    │   │   └── ImageLoader.m
    │   ├── navigation\
    │   │   ├── NavigationPlanner.m
    │   │   ├── Navigator.m
    │   │   ├── PathPlanner.m
    │   │   └── TerrainCost.m
    │   ├── robot\
    │   │   └── Robot.m
    │   └── vision\
    │       ├── OcclusionChecker.m
    │       └── VisionSystem.m
    └── ui\
        ├── InputHandler.m
        ├── Renderer.m
        └── SimulationUI.m
```

## Step-by-Step Setup

### Step 1: Replace SimConfig.m

Replace the current `SimConfig.m` file with the updated version that has the correct path:

**Location:** `src\config\SimConfig.m`

**Replace line 11 with:**
```matlab
DATASET_PATH = fullfile(fileparts(fileparts(fileparts(mfilename('fullpath')))), 'data', 'dataset')
```

This will automatically find the dataset folder relative to the SimConfig.m location.

### Step 2: Add startup.m to Root

Place `startup.m` in the root `files\` folder:

```
D:\...\files\
├── startup.m          ← ADD THIS FILE HERE
├── data\
├── docs\
├── output\
└── src\
```

### Step 3: Add Oracle Bone Images

Place your oracle bone images in the dataset folders:

```
data\dataset\
├── R1\             ← Add R1 scroll images (edges/corners only)
│   ├── image1.png
│   ├── image2.png
│   └── ...
├── real\           ← Add R2 real scroll images
│   ├── image1.png
│   ├── image2.png
│   └── ...
└── fake\           ← Add R2 fake scroll images
    ├── image1.png
    └── ...
```

## Running the Simulation

### Method 1: Using startup.m (RECOMMENDED)

1. Open MATLAB
2. Navigate to: `D:\Robotics Club\Robocon2026\...\files\`
3. In MATLAB Command Window, type:
   ```matlab
   startup
   ```
4. Press Enter

The startup script will:
- Add all necessary paths
- Verify installation
- Check dataset folders
- Launch the simulation automatically

### Method 2: Manual Setup

If you prefer manual control:

```matlab
% In MATLAB Command Window
cd('D:\Robotics Club\Robocon2026\Team-Vulcans-Robocon-2026\teams\team-2_vision\Matlab_Simulation\files');
addpath(genpath('src'));
robocon_arena_sim
```

### Method 3: Create Desktop Shortcut

Create `run_simulation.m` in the `files\` folder:

```matlab
% run_simulation.m
cd('D:\Robotics Club\Robocon2026\Team-Vulcans-Robocon-2026\teams\team-2_vision\Matlab_Simulation\files');
startup;
```

Then just double-click this file to run!

## Verification Checklist

Before running, verify:

- [ ] All 14 .m files are in correct folders
- [ ] `SimConfig.m` has updated path (line 11)
- [ ] `startup.m` is in root `files\` folder
- [ ] Dataset folders exist (even if empty)
- [ ] Oracle bone images are in `data\dataset\R1\`, `real\`, `fake\`

## Expected Console Output

When you run `startup`, you should see:

```
=== Robocon 2026 Simulation Startup ===

[1/3] Adding source folders to MATLAB path...
      ✓ All source folders added

[2/3] Verifying installation...
      ✓ All critical files found

[3/3] Checking dataset folder...
      ✓ Dataset folder exists: D:\...\files\data\dataset
      ✓ R1 folder: 3 images found
      ✓ Real folder: 4 images found
      ✓ Fake folder: 1 images found

=== Setup Complete ===

Starting simulation...

[CONFIG] Dataset path: D:\...\files\data\dataset
[INFO] Loading oracle bone character images...
[OK] Loaded 3 REAL, 1 FAKE oracle bone images
...
```

## Troubleshooting

### Issue 1: "Undefined function or variable 'robocon_arena_sim'"

**Cause:** MATLAB path not set up correctly

**Solution:**
```matlab
cd('D:\Robotics Club\Robocon2026\Team-Vulcans-Robocon-2026\teams\team-2_vision\Matlab_Simulation\files');
addpath(genpath('src'));
```

### Issue 2: "Dataset folder not found"

**Cause:** SimConfig.m has old hardcoded path

**Solution:** Update `SimConfig.m` line 11 with the new relative path code

### Issue 3: "No R1/R2/Fake images found, using placeholder"

**Cause:** Dataset folders are empty

**Solution:** This is OK! Simulation will use colored placeholders. Add images later.

### Issue 4: "Class not found" errors

**Cause:** Not all folders added to MATLAB path

**Solution:** 
```matlab
addpath(genpath('D:\Robotics Club\Robocon2026\Team-Vulcans-Robocon-2026\teams\team-2_vision\Matlab_Simulation\files\src'));
```

## Quick Test

To quickly test if everything is set up:

```matlab
% In MATLAB Command Window
cd('D:\Robotics Club\Robocon2026\Team-Vulcans-Robocon-2026\teams\team-2_vision\Matlab_Simulation\files');
addpath(genpath('src'));

% Test configuration
config = SimConfig();

% Should display dataset path
% If error, path needs fixing
```

## Files You Need to Update

1. **SimConfig.m** - Replace with updated version (automatic path detection)
2. **startup.m** - Add to root `files\` folder

## Next Steps After Installation

1. **Test the simulation:** Run `startup` and verify it launches
2. **Add oracle bone images:** Place images in dataset folders
3. **Try manual control:** Use W/A/S/D keys to move robot
4. **Enable autonomous mode:** Press 'N' key
5. **Read documentation:** Check `docs\README.md` for full controls

## Support

If you encounter issues:
1. Check `docs\QUICK_REFERENCE.md` for debugging tips
2. Verify folder structure matches exactly
3. Ensure MATLAB version is R2019b or later
4. Check MATLAB path includes all `src` subfolders

## Summary

Your folder structure is **perfect**! Just:
1. Replace `SimConfig.m` with updated version
2. Add `startup.m` to root folder
3. Run `startup` in MATLAB

The simulation will work even without images (uses placeholders).
