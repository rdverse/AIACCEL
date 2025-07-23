#
# @lc app=leetcode id=17 lang=python3
#
# [17] Letter Combinations of a Phone Number
#

# @lc code=start
class Solution:
    def letterCombinations(self, digits: str) -> List[str]:
        result = []
        if not digits:
            return result 

        # Map all the digits to their corresponding letters
        letters = {
            "2": "abc",
            "3": "def",
            "4": "ghi",
            "5": "jkl",
            "6": "mno",
            "7": "pqrs",
            "8": "tuv",
            "9": "wxyz",
        }
        
        def backtrack(i, cur_str):
            if len(cur_str) == len(digits):
                result.append(cur_str)
                return
             
            for c in letters[digits[i]]:
                backtrack(i+1, cur_str+c)
        
        if digits:
            backtrack(0, "")
        return result

         
# @lc code=end

