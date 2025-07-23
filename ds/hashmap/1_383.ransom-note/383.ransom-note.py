#
# @lc app=leetcode id=383 lang=python3
#
# [383] Ransom Note
#

# @lc code=start
from collections import Counter
class Solution:
    def canConstruct(self, ransomNote: str, magazine: str) -> bool:
        if ransomNote=="":
            return True
        
        #rmap = Counter(ransomNote)
        mag_map = Counter(magazine)

        for item in ransomNote:
            if item not in mag_map:
                return False
            if mag_map[item]<=0:
                return False 
            mag_map[item]-=1
        # O(m+n), #O(m) 
        return True
        
# @lc code=end

