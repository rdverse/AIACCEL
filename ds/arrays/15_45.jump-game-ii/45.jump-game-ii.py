#
# @lc app=leetcode id=45 lang=python3
#
# [45] Jump Game II
#

# @lc code=start
class Solution:
    def jump(self, nums: List[int]) -> int:
        # brute force with tree - O(n^n) and then find shortest path 
        # optimal solution
        # 
        """iterate from back - check if previous jump is better
            update only if the previous one is better
            [2 1 3 4]
        """  
        n = len(nums)
        pos, cap, dist = 0,0,0
        njumps=0
        # start at 0
        for i in range(0,n-1):
            dist = i + nums[i]
            # check capacity of cur indx
            cap = max(cap, dist)
             
            # update if capacity is limited  to a certain i
            if i==pos or i==0: 
               njumps+=1
               pos=cap
               
        return njumps 
             
# @lc code=end

