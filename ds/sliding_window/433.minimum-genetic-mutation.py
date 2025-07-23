#
# @lc app=leetcode id=433 lang=python3
#
# [433] Minimum Genetic Mutation
#

# @lc code=start
import collections
class Solution:
    def minMutation(self, startGene: str, endGene: str, bank: List[str]) -> int:
        choices = set(("A", "C", "G", "T"))
        bank = set(bank)
        queue = collections.deque([startGene, 0]) 
        visited = {startGene} 
        while queue:
            (node, steps) = queue.popleft()
            if node == endGene:
                return steps
            for c in "AGCT":
                for i in range(len(node)):
                    neighbor = node[:i] + c + node[i+1:]
                    if neighbor not in visited and neighbor in bank:
                        queue.append((neighbor, steps+1))
                        visited.add(neighbor)
        return -1 # O(Bank), O(1)
                        
# @lc code=end

