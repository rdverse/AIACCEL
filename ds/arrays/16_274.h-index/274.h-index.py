#
# @lc app=leetcode id=274 lang=python3
#
# [274] H-Index
#

# @lc code=start
class Solution:
    def hIndex(self, citations: List[int]) -> int:
        # naive is O(n^2) as we need to count for each number
        # create a hashmap with the count of each h-index
        # starting from the max, iteratively step back (npapers_c<citations) to find max h-index
        citations_count = {} # ,O(n)
        n = len(citations) # ,O(1)
        for citation in citations: # O(n),
            citation = min(citation, n)
            if citation in citations_count:
                citations_count[citation] += 1
            else:
                citations_count[citation] = 1  
        
        h_index = 0
        rolling_sum =0
        for i in range(n,-1,-1): # O(n),
            if i in citations_count: # O(1),
                rolling_sum+=citations_count[i]
                if rolling_sum>=i:
                    h_index= max(i,h_index) 
                else:
                    h_index = max(rolling_sum ,h_index)
        return h_index 
        
# @lc code=end

