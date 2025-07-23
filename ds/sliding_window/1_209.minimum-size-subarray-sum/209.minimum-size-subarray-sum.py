#
# @lc app=leetcode id=209 lang=python3
#
# [209] Minimum Size Subarray Sum
#
# @lc code=start
class Solution:
    def minSubArrayLen(self, target: int, nums: List[int]) -> int:
        l, total = 0, 0
        minlength =  len(nums)+10
        for r in range(len(nums)):
            total += nums[r]
            
            while total >= target:
                minlength = min(r-l+1, minlength)
                total -= nums[l]
                l += 1 
        
        return 0 if minlength==(len(nums)+10) else minlength # O(2n), O(1)
# @lc code=end