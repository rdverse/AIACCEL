#
# @lc app=leetcode id=76 lang=python3
#
# [76] Minimum Window Substring
#

# @lc code=start
from collections import Counter

class Solution:
    def minWindow(self, s: str, t: str) -> str:
        # O(m+n), O(n)
        if t == "": return ""
        
        tdict = Counter(t)  # Required character counts
        window = {}         # Current window character counts
        have, need = 0, len(tdict)  # have: # of chars meeting requirement, need: total unique chars in t
        res, resLen = [-1, -1], float('inf')
        l = 0

        for r, c in enumerate(s):
            window[c] = window.get(c, 0) + 1
            if c in tdict and window[c] == tdict[c]:
                have += 1
            while have == need:
                # Update result if this window is smaller
                if (r - l + 1) < resLen:
                    res = [l, r]
                    resLen = r - l + 1
                # Pop from left
                window[s[l]] -= 1
                if s[l] in tdict and window[s[l]] < tdict[s[l]]:
                    have -= 1
                l += 1
        l, r = res
        return s[l:r+1] if resLen != float('inf') else ""
            
            
             
             
# @lc code=end

