#
# @lc app=leetcode id=26 lang=python3
#
# [26] Remove Duplicates from Sorted Array
#

# @lc code=start
class Solution:
    def removeDuplicates(self, nums: List[int]) -> int:
        # [1,2,1,1]
        #       j i
        # {1:0,2:1}
        # naive
        # {k:v for k,v}
        # O(n) + O(n)
        # optimized : O(n), O(1)
        j=1;i=1
        n=len(nums)
        while i<n:
            if nums[i]!=nums[i-1]:
                nums[j]=nums[i]
                j+=1
            i+=1
        # j keeps a count of all the unique nos
        return j
        
# @lc code=end

