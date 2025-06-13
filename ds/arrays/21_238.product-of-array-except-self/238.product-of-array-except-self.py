#
# @lc app=leetcode id=238 lang=python3
#
# [238] Product of Array Except Self
#

# @lc code=start
class Solution:
    def productExceptSelf(self, nums: List[int]) -> List[int]:
       # [x1, x2]
       # [x2, x1] 
       
       # [x1, x2, x3]
       # [x3*x2, x1*x2, x3*x1]
       
       # [x1, x2, x3, x4]
       # [x2*x3*x4, x1*x3*x4, x2*x1*x4, x2*x3*x1]
      
       # sol 1->1-n
       # [1, x1*1, x1*1*x2, x1*1*x2*x3]
       # n-1 
       # R=1*x4*x3
       # [x2*1*x4*x3, x1*1*R(1*x4*x3), x1*1*x2*R(1*x4), x1*1*x2*x3]
       # i=1,n
        # O(2n) , O(1) [O(n)-result]
        n = len(nums) 
        result = [0]*n
        result[0] = 1
        for i in range(1, n):
            result[i] = result[i-1] * nums[i-1]
        
        R = 1
        for i in range(n-1, -1, -1):
            result[i] *= R 
            R *= nums[i]
        
        return result


        
# @lc code=end

