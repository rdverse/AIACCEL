#
# @lc app=leetcode id=290 lang=python3
#
# [290] Word Pattern
#

# @lc code=start
class Solution:
    def wordPattern(self, 
                    pattern: str,
                    s: str) -> bool:
        
        words = s.split() # O(L)
        chars = list(pattern)
        library = {} # char : word
         
        if len(words)!=len(chars):
            return False
        # "abba"
        # "dog cat cat fish"
        # 0 : a, dog 
        # 1 : b, cat
        # 2 : b, cat
        # 3 :  
        for id,[word, char] in enumerate(list(zip(words, chars))):
            char_key = f"char_{char}"
            word_key = f"word_{word}"
            
            if char_key not in library:
                library[char_key]=id
            
            if word_key not in library:
                library[word_key]=id

            if library[word_key]!= library[char_key]:
                return False 
            
        return True 
         
        
# @lc code=end

