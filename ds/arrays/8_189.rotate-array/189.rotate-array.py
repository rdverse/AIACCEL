#
# @lc app=leetcode id=189 lang=python3
#
# [189] Rotate Array
#

# @lc code=start
class Solution:
    def rotate(self, nums: List[int], k: int) -> None:
        """
        Do not return anything, modify nums in-place instead.
        """
        # [1,2,3,4,5,6,7]
        # [6,  7,  1,  2,  3,  4,  5]
        # 5-0 6-1 0-2  1-3 2-4 3-5 4-6
        # [5,  6,  7,  1,  2,  3,  4]
        #  4-0 5-1 6-2 0-3 1-4 2-5 3-6
        # to move 1 - 0-3, 3-6, 6-2, 2-5... 
        # insert each el - O(nxk)
        # clip, reverse, append

        # iterate over loop - curr and prev | i : [0,6], k:R , n : len(nums)  
        # indek = (i+k) ? i+k<n-1 : (i+k)%(n-1)

        # k=3, n=7
        # n total replacements
        # first move 1 to 4, 5 to 7 ....  
        # [-1,  -100,  3,  99]
        # [ 3,  99,  -1,  -100] 
        #  2-0  3-1   0-2  1-3
        
        n = len(nums)
        prev_idx = 0
        count = n
        prev_item = nums[0] 
        start = 0
        while(count>0):
            prev_idx = start
            prev_item = nums[start]
            while(True):
                #prev_item = nums[prev_idx]
                curr_idx = (prev_idx+k)%(n)

                curr_item = nums[curr_idx]
                nums[curr_idx] = prev_item
                prev_item = curr_item
                prev_idx = curr_idx 
                
                count-=1
                if start==curr_idx:
                    break
            start+=1 
        # [-1, -100, 3, 99]
        # k=2
        # prev_idx  : -1
        # prev_item : 0 
        # curr_idx  : 2+2=4%4=0 
        # curr_item : -1

        # prev_item : 4
        # prev_idx  : 3 
        
# @lc code=end

