#
# @lc app=leetcode id=169 lang=python3
#
# [169] Majority Element
#

# @lc code=start
class Solution:
    def majorityElement(self, nums: List[int]) -> int:
        # O(n), O(1) - knowing candidate is occuring n/2 times is important
        # other methods : hashmap, divide&conquer, sort, brite force
        count=0;candidate=None 
        for num in nums:
            if count==0:
                candidate=num
            count+=1 if num==candidate else -1
        return candidate
        
# @lc code=end

