#
# @lc app=leetcode id=58 lang=python3
#
# [58] Length of Last Word
#

# @lc code=start
class Solution:
    def lengthOfLastWord(self, s: str) -> int:
        parse = False 
        count = 0
        for i in range(len(s),-1,-1):
            if s[i]==" " and not parse:
               continue
            elif s[i]==" " and parse:
                break
            else:
                count+=1
                parse = True
                
        return count 
       # easy sol : return 0 if not s or s.isspace() else len(s.split()[-1])
   
# @lc code=end

