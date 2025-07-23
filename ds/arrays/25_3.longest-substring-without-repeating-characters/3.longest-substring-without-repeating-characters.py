#
# @lc app=leetcode id=3 lang=python3
#
# [3] Longest Substring Without Repeating Characters
#

# @lc code=start
class Solution:
    def lengthOfLongestSubstring(self, s: str) -> int:
        # sliding window 
        charset = set()
        max_substr = 0
        l=0
        for r in range(len(s)):
            while s[r] in charset:
                # remove char from charset if present
                charset.remove(s[l])
                # move l as we removed
                l+=1
            charset.add(s[r])
            max_substr = max(max_substr, r-l+1)
        return max_substr
            # O(n), O(k)- size of charset
            
               
               
               
       
       
       
       
       
        
# @lc code=end

