classdef OcclusionChecker
    % OCCLUSIONCHECKER Utility for checking if KFS are occluded by blocks
    
    methods (Static)
        function occluded = isOccluded(p1, p2, blocks, blockSize, targetId)
            % Check if line of sight from p1 to p2 is blocked
            occluded = false;
            
            for k = 1:numel(blocks)
                if blocks(k).id == targetId
                    continue;
                end
                
                b = blocks(k);
                minB = [b.x - blockSize/2, b.y - blockSize/2, 0];
                maxB = [b.x + blockSize/2, b.y + blockSize/2, b.h + 50];
                
                if OcclusionChecker.rayBoxIntersect(p1, p2, minB, maxB)
                    occluded = true;
                    return;
                end
            end
        end
        
        function hit = rayBoxIntersect(p1, p2, minB, maxB)
            % Ray-box intersection test (slab method)
            dir = p2 - p1;
            tmin = 0;
            tmax = 1;
            
            for i = 1:3
                if abs(dir(i)) < 1e-6
                    if p1(i) < minB(i) || p1(i) > maxB(i)
                        hit = false;
                        return;
                    end
                else
                    invD = 1.0 / dir(i);
                    t1 = (minB(i) - p1(i)) * invD;
                    t2 = (maxB(i) - p1(i)) * invD;
                    
                    if t1 > t2
                        tmp = t1;
                        t1 = t2;
                        t2 = tmp;
                    end
                    
                    tmin = max(tmin, t1);
                    tmax = min(tmax, t2);
                    
                    if tmin > tmax
                        hit = false;
                        return;
                    end
                end
            end
            
            hit = true;
        end
    end
end
