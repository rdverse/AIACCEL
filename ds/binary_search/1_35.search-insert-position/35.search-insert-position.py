#
# @lc app=leetcode id=35 lang=python3
#
# [35] Search Insert Position
#

# @lc code=start
class Solution:
    def searchInsert(self, nums: List[int], target: int) -> int:
      
        left, right = 0, len(nums)-1   
      

        while left<=right:
            pivot =  (left+right)//2
            if target==nums[pivot]:
                return pivot 
            if target < nums[pivot]:
                right = pivot -1
            else:
                left  =pivot+1
            
        return left  # O(logn)
        
# @lc code=end

