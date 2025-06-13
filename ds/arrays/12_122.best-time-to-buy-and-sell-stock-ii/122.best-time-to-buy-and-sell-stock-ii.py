#
# @lc app=leetcode id=122 lang=python3
#
# [122] Best Time to Buy and Sell Stock II
#

# @lc code=start
class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        buy = prices[0]
        total_profit = 0
        for i in range(0, len(prices)-1):
            sell = prices[i+1]
            new_profit = sell-buy
            if new_profit>0:
                total_profit+=new_profit
                buy=sell
            else:
                buy=sell 
       
        return total_profit # O(n), O(1)
       
       
# @lc code=end

