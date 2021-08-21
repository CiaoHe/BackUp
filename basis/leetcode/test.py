class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

# def pass(head_node):
#     p = head_node
#     while p:
#         print(p.val)
#         p = p.next

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

def reverse(a:ListNode, b:ListNode):
    if a == b:
        return a
    successor = b.next
    newhead = reverse(a.next,b)
    a.next.next = a
    a.next = successor
    return newhead

link = list2link([1,2,3,4,5,8])
printall(reverse(link, link.next.next))