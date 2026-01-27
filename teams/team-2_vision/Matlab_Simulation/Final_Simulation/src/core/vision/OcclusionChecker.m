classdef OcclusionChecker
    % ===================================================================
    % OCCLUSIONCHECKER
    % -------------------------------------------------------------------
    % Static utility class responsible for determining whether a straight
    % line of sight between two 3D points is blocked by any obstacle blocks.
    %
    % This models physical occlusion realistically using ray casting.
    %
    % Core responsibilities:
    %   - Cast a ray from camera position to target point.
    %   - Test intersection against every block bounding box.
    %   - Report whether any obstacle blocks the view.
    %
    % Mathematical foundation:
    %   - Uses Axis-Aligned Bounding Box (AABB) intersection.
    %   - Implements the classic "slab method" for ray-box intersection.
    %
    % This class contains no state and no side effects.
    % It behaves as a deterministic geometric oracle.
    % ===================================================================
    
    methods (Static)
        
        % ===================================================================
        % Line-of-Sight Occlusion Test
        % -------------------------------------------------------------------
        % Determines whether the ray segment from p1 to p2 intersects any
        % obstacle block.
        %
        % Inputs:
        %   p1        → Ray origin (camera position) [x y z]
        %   p2        → Ray target (object center)   [x y z]
        %   blocks    → Array of block structs in arena
        %   blockSize → Footprint dimension of blocks (XY size)
        %   targetId  → Block ID of the target (ignored for occlusion)
        %
        % Output:
        %   occluded  → true if any obstacle blocks the line of sight
        %
        % Algorithm:
        %   - For each block (except target):
        %       - Construct its axis-aligned bounding box.
        %       - Check if ray intersects the box.
        %       - Early exit if intersection found.
        % ===================================================================
        function occluded = isOccluded(p1, p2, blocks, blockSize, targetId)
            
            % Default assumption: visibility is clear
            occluded = false;
            
            % Iterate over every block in the arena
            for k = 1:numel(blocks)
                
                % Skip the block that contains the target itself
                if blocks(k).id == targetId
                    continue;
                end
                
                % Extract block structure
                b = blocks(k);
                
                % -----------------------------------------------------------
                % Construct axis-aligned bounding box (AABB) for block.
                %
                % minB → Lower corner of the box
                % maxB → Upper corner of the box
                %
                % The +50 offset in height provides a small safety margin
                % to prevent grazing edge misses.
                % -----------------------------------------------------------
                minB = [ ...
                    b.x - blockSize/2, ...
                    b.y - blockSize/2, ...
                    0 ];
                
                maxB = [ ...
                    b.x + blockSize/2, ...
                    b.y + blockSize/2, ...
                    b.h + 50 ];
                
                % -----------------------------------------------------------
                % Perform ray–box intersection test.
                % -----------------------------------------------------------
                if OcclusionChecker.rayBoxIntersect(p1, p2, minB, maxB)
                    occluded = true;
                    return;     % Early exit for efficiency
                end
            end
        end
        
        
        % ===================================================================
        % Ray–Box Intersection Test (Slab Method)
        % -------------------------------------------------------------------
        % Determines whether the ray segment from p1 to p2 intersects the
        % axis-aligned bounding box defined by minB and maxB.
        %
        % Mathematical model:
        %   - Ray is parameterized as:
        %         R(t) = p1 + t * (p2 - p1),  t ∈ [0, 1]
        %
        %   - Intersection occurs if the ray overlaps all three axis slabs.
        %
        % Inputs:
        %   p1    → Ray origin
        %   p2    → Ray endpoint
        %   minB  → Minimum corner of bounding box
        %   maxB  → Maximum corner of bounding box
        %
        % Output:
        %   hit   → true if intersection exists
        % ===================================================================
        function hit = rayBoxIntersect(p1, p2, minB, maxB)
            
            % Direction vector of the ray
            dir = p2 - p1;
            
            % Valid ray parameter interval
            tmin = 0;
            tmax = 1;
            
            % ---------------------------------------------------------------
            % Process intersection independently for X, Y, Z axes
            % ---------------------------------------------------------------
            for i = 1:3
                
                % -----------------------------------------------------------
                % If ray is nearly parallel to slab planes
                % -----------------------------------------------------------
                if abs(dir(i)) < 1e-6
                    
                    % If origin lies outside slab, no intersection possible
                    if p1(i) < minB(i) || p1(i) > maxB(i)
                        hit = false;
                        return;
                    end
                    
                else
                    % -------------------------------------------------------
                    % Compute intersection parameters along this axis
                    % -------------------------------------------------------
                    invD = 1.0 / dir(i);
                    
                    t1 = (minB(i) - p1(i)) * invD;
                    t2 = (maxB(i) - p1(i)) * invD;
                    
                    % Ensure t1 <= t2
                    if t1 > t2
                        tmp = t1;
                        t1  = t2;
                        t2  = tmp;
                    end
                    
                    % Narrow valid interval
                    tmin = max(tmin, t1);
                    tmax = min(tmax, t2);
                    
                    % If interval collapses, no intersection exists
                    if tmin > tmax
                        hit = false;
                        return;
                    end
                end
            end
            
            % If all slabs overlap, intersection exists
            hit = true;
        end
    end
end
