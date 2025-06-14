#
# @lc app=leetcode id=141 lang=python3
#
# [141] Linked List Cycle
#

# @lc code=start
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution:
    def hasCycle(self, head: Optional[ListNode]) -> bool:
        if not head:
            return False
       
        slow = head
        fast = head.next
        
        while(True): # if there is a cycle this will end
            
            if slow==None or fast==None:
                return False
            
            if slow==fast:
                return True
            
            else:
               slow = slow.next
               fast = fast.next if fast.next==None else fast.next.next
               
            # O(n), O(1)
# @lc code=end

