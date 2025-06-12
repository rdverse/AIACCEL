#
# @lc app=leetcode id=88 lang=python3
#
# [88] Merge Sorted Array
#

# @lc code=start
class Solution:
    def merge(self, nums1: List[int], m: int, nums2: List[int], n: int) -> None:
        """
        Do not return anything, modify nums1 in-place instead.
        """
        # naive solution will have O(mxn)
        # this takes O(m+n) as we iterate through m+n array
        # space = O(m+n)
        i = m - 1
        j = n - 1 
        for k in range(m+n-1, -1,-1):
            if i<0 and j>=0:
                nums1[k] = nums2[j]
                j-=1
            elif j<0 and i>=0:
                nums1[k] = nums1[i]
                i-=1
            elif nums1[i]>nums2[j]:
                nums1[k]=nums1[i]
                i-=1
            else:
                nums1[k]=nums2[j]
                j-=1
            
                
         
# @lc code=end

