class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

# 输入：(2 -> 4 -> 3) + (5 -> 6 -> 4)
# 输出：7 -> 0 -> 8
class Solution:
    def addTwoNumbers(self, l1: ListNode, l2: ListNode) -> ListNode:
        dummy = ListNode(-1) #虚拟头节点
        cur = dummy #pointer to dummy (开始先对准dummy,结束直接输出dummy.next)
        t = 0 #进位
        while (l1 or l2 or t):
            if l1:
                t+= l1.val
                l1 = l1.next
            if l2:
                t+=l2.val
                l2 = l2.next
            cur.next = ListNode(t%10)
            #记得挪cur
            cur = cur.next
            t = t//10
        return dummy.next

def create(l):
    dummy = ListNode(-1)
    cur = dummy
    for i in range(len(l)):
        cur.next = ListNode(l[i])
        cur = cur.next
    return dummy.next

def show(link):
    while link:
        print(link.val)
        link = link.next

s = Solution()
l1 = create([2,4,3])
l2 = create([5,6,4])
show(s.addTwoNumbers(l1,l2))