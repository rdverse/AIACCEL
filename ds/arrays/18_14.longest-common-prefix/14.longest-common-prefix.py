#
# @lc app=leetcode id=14 lang=python3
#
# [14] Longest Common Prefix
#

# @lc code=start
class Solution:
    # vertical scanning O(s) s = sum(strings in list), O(1)
    # other methods include using trie
    def longestCommonPrefix(self, strs: List[str]) -> str:
        if len(strs)==1:
            return strs[0]
        elif len(strs)==0 or strs==None:
            return ""
        common_prefix = ""
        pointer = 0
        while True: # O(n) - n : no of strings 
            for i in range(0,len(strs)-1): # O(m) : length of smallest string
                if pointer>(len(strs[i])-1) or pointer>(len(strs[i+1])-1) or strs[i]=="" or strs[i+1]=="":
                    return common_prefix
                elif strs[i][pointer]!=strs[i+1][pointer]:
                    return common_prefix
                else:
                    continue 
                
            common_prefix+=strs[i][pointer]
            pointer+=1
            
# @lc code=end

