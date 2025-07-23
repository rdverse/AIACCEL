#
# @lc app=leetcode id=55 lang=python3
#
# [55] Jump Game
#
# Brute force -  building a tree of all possible solutions O(n^n)
# use dynamic programming
class Solution:
    def canJump(self, nums: List[int]) -> bool:
        position = len(nums) -1
        for idx in range(len(nums)-1, -1,-1):
            if idx + nums[idx]>=position:
                #[1 0 3 4]
                #[1 2 3 4]
                position = idx
        return position==0 
                  
# @lc code=end

