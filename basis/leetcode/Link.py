class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

def list2link(l):
    head = ListNode(-1)
    p = head
    for i in l:
        new = ListNode(i)
        p.next = new
        p = p.next
    return head.next

def printall(head):
    while head:
        print(head.val)
        head = head.next