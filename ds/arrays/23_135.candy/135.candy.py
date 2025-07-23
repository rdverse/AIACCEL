#
# @lc app=leetcode id=135 lang=python3
#
# [135] Candy
#

# @lc code=start
class Solution:
    def candy(self, ratings: List[int]) -> int:
        n = len(ratings)
        ncandy = [1]*n
        # forward pass
        for i in range(0,n-1):
            if ratings[i]<ratings[i+1]:
                ncandy[i+1] = ncandy[i]+1
        # [1,2>3]
        for i in range(n-1,0,-1):
            if ratings[i]<ratings[i-1]:
                ncandy[i-1] = max(ncandy[i-1],ncandy[i]+1)

        return sum(ncandy) 
            
             
        
# @lc code=end

