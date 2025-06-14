#
# @lc app=leetcode id=134 lang=python3
#
# [134] Gas Station
#

# @lc code=start
class Solution:
    def canCompleteCircuit(self, gas: List[int], cost: List[int]) -> int:
        # net gas 
        # start
        # total 
        n = len(gas) 
        start = 0
        total = 0
        diff = 0
        tank = 0
        for i in range(n):
            diff = gas[i] - cost[i]
            total += diff 
            tank += diff
              
            if tank<0:
               start = i+1
               tank = 0
               
        return start if total>=0 else -1

            
# @lc code=end

