#
# @lc app=leetcode id=27 lang=python3
#
# [27] Remove Element
#

# @lc code=start
class Solution:
    def removeElement(self, nums: List[int], val: int) -> int:
        # 2 pointers approach
        # O(n), O(n)
        #[ 1, 2, 1, 2, 2], 2
        #|
        #i,j
        # too many replacements for below sol even if O(n), O(1)
        # k =0 
        # j = 0 
        # for i in range(len(nums)):
        #     if nums[i] != val:
        #         nums[j]=nums[i]
        #         j+=1

        #     return k
        i=0
        n=len(nums)
        while(i<n):
            if nums[i]==val:
                nums[i]=nums[n-1]
                n-=1
            else:
                i+=1
        return i 
                
        
# @lc code=end

