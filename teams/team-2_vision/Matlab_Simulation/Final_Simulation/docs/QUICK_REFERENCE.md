# Quick Reference Guide

## Component Interaction Flow

### Simulation Loop
```
Main Loop:
  1. VisionSystem.processFrame() → detections
  2. Navigator.step() (if autonomous)
  3. SimulationUI.render()
  4. drawnow
```

### Vision Detection Flow
```
VisionSystem.processFrame():
  For each KFS:
    - Check distance (DETECTION_RANGE)
    - Check FOV angle (DETECTION_FOV)
    - Check occlusion (OcclusionChecker)
    - Check color (HSV filter)
    - Confirm over time (MIN_DETECTION_TIME)
    - Register detection
  Update grid belief
```

### Navigation Step Flow
```
Navigator.step():
  1. Update belief from vision
  2. Check time limits
  3. Try immediate KFS pickup
  4. Check if at exit
  5. Handle R1 waiting (if in WAIT mode)
  6. Select target:
     - COLLECT: chooseBestR2() or findBestExploreCell()
     - EXIT: planExitRoute()
  7. Execute move via PathPlanner
```

### A* Pathfinding Flow
```
PathPlanner.astarWithTerrain():
  1. Initialize open list, gScore, fScore
  2. While open list not empty:
     - Get lowest fScore cell
     - If goal reached, reconstruct path
     - For each neighbor:
       * Evaluate traversability
       * Calculate terrain cost
       * Add mode-specific penalties
       * Update scores if better
  3. Return path or empty
```

## Key Data Structures

### Grid Coordinates
- **[row, col]**: Grid cell position (1-indexed)
- **Row**: Increases from back to front (1 to FOREST_ROWS=4)
- **Col**: Increases from left to right (1 to FOREST_COLS=3)

### 3D Coordinates
- **[x, y, z]**: Millimeters in world space
- **x**: Left-right (0 to ARENA_X=6000)
- **y**: Back-front (0 to ARENA_Y=7300)
- **z**: Height (0 to ARENA_Z=2000)

### Block Structure
```matlab
block.id   % Unique identifier (1-12)
block.row  % Grid row (1-4)
block.col  % Grid column (1-3)
block.x    % Center X in mm
block.y    % Center Y in mm
block.h    % Height in mm (200/400/600)
```

### Belief Map
```matlab
belief(row, col) = "R2" | "R1" | "FAKE" | "EMPTY" | "UNSEEN" | "UNKNOWN"
beliefConfidence(row, col) = 0.0 to 1.0
```

## Common Tasks

### Add New Parameter
1. Open `SimConfig.m`
2. Add to `properties (Constant)` section
3. Use via `config.PARAMETER_NAME`

### Modify Vision Range
```matlab
% In SimConfig.m
DETECTION_RANGE = 3000  % Change this value
```

### Change Navigation Weights
```matlab
% In SimConfig.m
ALPHA0 = 3.0   % Weight for R2 when empty
BETA0 = 0.5    % Weight for exit when empty
ALPHA1 = 0.5   % Weight for R2 when full
BETA1 = 3.0    % Weight for exit when full
```

### Add Custom Rendering
```matlab
% In Renderer.m, add new static method
function drawCustomElement(ax, params, config)
    % Your drawing code here
end

% Call from SimulationUI.renderOverview()
Renderer.drawCustomElement(obj.axMain, params, cfg);
```

### Add Keyboard Shortcut
```matlab
% In InputHandler.handleKey()
case 'mykey'
    % Your action here
    fprintf('[MY FEATURE] activated\n');
```

### Modify Path Cost
```matlab
% In TerrainCost.move()
function c = move(cur, nxt, hMap, config)
    % Add custom cost logic
    baseCost = ...;
    customPenalty = ...;
    c = baseCost + customPenalty;
end
```

## Debugging Tips

### Enable Debug Prints
```matlab
% Add at start of problematic method
fprintf('[DEBUG] Variable: %s\n', var);
```

### Visualize Belief State
```matlab
% In Navigator or SimulationUI
figure; imagesc(beliefConfidence);
colorbar; title('Belief Confidence');
```

### Check Path Validity
```matlab
% After PathPlanner.astarWithTerrain()
if isempty(path)
    fprintf('[DEBUG] No path found from (%d,%d) to (%d,%d)\n', ...
        start, goal);
end
```

### Monitor Navigation State
```matlab
% In Navigator.step()
fprintf('[NAV] Mode: %s, Carry: %d/%d, Pos: (%d,%d)\n', ...
    obj.mode, obj.carry, cfg.CAPACITY, obj.position);
```

### Trace Detection
```matlab
% In VisionSystem.processFrame()
fprintf('[VISION] KFS %d: dist=%.0f, angle=%.1f, visible=%d\n', ...
    kfsId, dist, angle, isVisible);
```

## Performance Optimization

### Reduce Rendering Load
```matlab
% In SimulationUI.render(), add conditional rendering
if mod(frameCount, 5) == 0  % Render every 5 frames
    obj.renderCameraView(...);
end
```

### Limit A* Search
```matlab
% In PathPlanner, already limited to maxIterations
maxIterations = ROWS * COLS * 4;  % Reduce multiplier if needed
```

### Cache Expensive Calculations
```matlab
% In any class
persistent cachedValue;
if isempty(cachedValue)
    cachedValue = expensiveComputation();
end
```

## Common Errors and Fixes

### "Index exceeds array bounds"
- **Cause**: Grid coordinates out of range
- **Fix**: Check `PathPlanner.isValidCell()` before accessing arrays

### "Key not found in Map"
- **Cause**: Trying to access non-existent detection
- **Fix**: Use `isKey()` before accessing Maps

### "NaN in robot position"
- **Cause**: Invalid movement calculation
- **Fix**: Add bounds checking in `Robot.move()`

### "Empty path returned"
- **Cause**: No valid path exists or all paths blocked
- **Fix**: Check obstacle placement and cost penalties

### "Stuck counter incrementing"
- **Cause**: Robot not moving or moving minimally
- **Fix**: Increase `POSITION_TOLERANCE` or check movement logic

## Testing Checklist

- [ ] Robot movement (W/A/S/D/Q/E)
- [ ] Block climbing (SPACE) and descending (C)
- [ ] Camera controls (I/K/Z/X/V)
- [ ] Vision detection (blue KFS within range)
- [ ] Autonomous navigation (N key)
- [ ] KFS randomization (T key)
- [ ] Path recording (P key)
- [ ] Data export (L key)
- [ ] Reset functionality (R key)
- [ ] Collision detection (arena bounds, forest area)
- [ ] R1 waiting behavior
- [ ] Exit route planning
- [ ] Time-based emergency exit

## File Modification Priority

**High Impact** (core functionality):
- Navigator.m
- PathPlanner.m
- SimConfig.m

**Medium Impact** (behavior tuning):
- NavigationPlanner.m
- TerrainCost.m
- VisionSystem.m

**Low Impact** (user experience):
- SimulationUI.m
- Renderer.m
- InputHandler.m

**Rarely Modified**:
- Arena.m
- Robot.m
- OcclusionChecker.m
- ImageLoader.m
