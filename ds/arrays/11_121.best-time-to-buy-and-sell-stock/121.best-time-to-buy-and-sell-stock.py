#
# @lc app=leetcode id=121 lang=python3
#
# [121] Best Time to Buy and Sell Stock
#

# @lc code=start
class Solution:
    def maxProfit(self, prices: List[int]) -> int:
       # each item is price of stock on a particular day
       # maximize profit
       # naive approach is O(n^2) - all combinations
       # [7,1,5,3,6,4]
       # []
       # max is negative?
        max_profit=0
        buy = prices[0] 
        for i in range(len(prices)-1):
            sell = prices[i+1]
            new_profit = sell-buy
            if new_profit > max_profit:
                max_profit = new_profit
            elif new_profit<0:
                buy = sell
        return max_profit
        
        
# @lc code=end

