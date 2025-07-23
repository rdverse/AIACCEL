#
# @lc app=leetcode id=13 lang=python3
#
# [13] Roman to Integer
#

# @lc code=start

VALUES = {"I":1, "V":5, "X": 10, "L":50, "C":100,"D":500,"M":1000}
class Solution:
    def romanToInt(self, s: str) -> int:
        total_sum = 0
        for i in range(1,len(s)+1):
            if i<len(s) and VALUES[s[i-1]]<VALUES[s[i]]:
                total_sum -= VALUES[s[i-1]]
            else:
                total_sum += VALUES[s[i-1]]
        return total_sum # O(n) + O(m) for the map, O(1) for the program
# @lc code=end

