#
# @lc app=leetcode id=80 lang=python3
#
# [80] Remove Duplicates from Sorted Array II
#

# @lc code=start
class Solution:
    def removeDuplicates(self, nums: List[int]) -> int:
        #[0,0,0,1,2,3,1,1,3]
        #   l   r 
        #    c=3
        #[0,0,1,1,2,3,3] k = 7
        # [1, 1, 1, 2, 2, 3]
        #      l     r
        #            c=3
        # popping elements takes O(n^2) at worst due to the need for popping which is O(n)
        # optimized : O(n), O(1)
        if not nums:
            return 0
        l=1;r=1;count=1
        while(r<len(nums)):
            if nums[r-1]==nums[r]:
                count+=1
                if count>2:
                    r+=1
                    continue
            else:
                count=1
            nums[l] = nums[r]
            l+=1
            r+=1

        # j keeps a count of all the unique nos
        return l

# @lc code=end

