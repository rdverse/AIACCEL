#
# @lc app=leetcode id=200 lang=python3
#
# [200] Number of Islands
#
import collections
# @lc code=start
class Solution:
    def numIslands(self, grid: List[List[str]]) -> int:
        nislands = 0
        if not grid:
            return nislands
        rows, cols = len(grid), len(grid[0])
        visited = set()
       
        def bfs(r,c):
            q = collections.deque()
            visited.add((r,c))
            q.append((r,c)) 
            
            while q:
                row, col = q.popleft()
                directions = [[1,0], 
                              [-1, 0], 
                              [0,1], 
                              [0,-1]]
                
                for dr, dc in directions:
                    r = row+dr
                    c = col+dc
                    if (r in range(rows) and 
                        c in range(cols) and 
                        grid[r][c]=="1" and
                        (r, c) not in visited):
                        q.append((r,c))
                        visited.add((r,c))
                          
        for r in range(rows):
            for c in range(cols):
                if grid[r][c] == "1":
                    bfs(r,c)
                    nislands+=1
                    
        return nislands 
        
# @lc code=end

