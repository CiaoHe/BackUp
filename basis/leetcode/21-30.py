#22
def generationParenthesis(n:int)->List(str):
#（1）任意前缀里，左括号数量>=右括号数量
#（2）左右括号数量相等
# 数量：卡特兰数 C(n,2n)/(n+1)
    #用dfs搜索树
    res = []
    dfs(n,0,0,'')

    def dfs(n,lc,rc,seq):
        if (lc==n and rc==n):
            res.append(seq)
        else:
            if lc<n:
                dfs(n,lc+1,rc,seq+'(')
            if rc<n and lc>rc:
                dfs(n,lc,rc+1,seq+')')
    return res

#23
'''
用最小堆来找K个链表的最小值
'''
import heapq
def mergeKLists(self, lists: List[ListNode]) -> ListNode:
    if len(lists)==0: return None
    if len(lists)==1: return lists[0]
    dummy = ListNode(-1,None)
    cur = dummy
    heap = []
    #init heap
    for i in range(len(lists)):
        if lists[i]:
            heapq.heappush(heap, lists[i].val)
            lists[i] = lists[i].next
    #loop while heap empty
    while heap:
        new = ListNode(heapq.heappop(heap))
        cur.next = new
        cur = new
        for i in range(len(lists)):
            if lists[i]:
                heapq.heappush(heap, lists[i].val)
                lists[i] = lists[i].next
    return dummy.next

#24 两两交换链表的元素
def swapPairs(self, head: ListNode) -> ListNode:
    dummy = ListNode(-1,head)
    p = dummy
    if p.next == None or p.next.next == None: return p.next
    while p.next and p.next.next:
        u1,u2,tail = p.next, p.next.next,p.next.next.next
        p.next = u2
        u2.next = u1
        u1.next = tail
        p = u1
    return dummy.next

#25.K 个一组翻转链表
'''
(1) 翻转以a Node开头的链表
'''
def reverse(ListNode a):
    prev = None
    cur = nxt = a
    while cur:
        nxt = cur.next
        cur.next = prev
        prev = cur
        cur = nxt
    return prev

'''
(2) 反转a到b之间的节点,(b不是翻转中的一部分，b是另一段)
'''
def reverse(ListNode a, ListNode b):
    prev = None
    cur=nxt=a
    while cur!=b:
        nxt = cur.next
        cur.next = prev
        prev = cur
        cur = nxt
    return 

'''
先反转前K个节点，K+1个节点往后递归翻转，然后接上分开的两段
'''
def reverseKGroup(ListNode head, k):
    a = b = head
    for i in range(k):
        if not b:
            return head
        b = b.next
    newhead = reverse(a,b)
    a.next = reverseKGroup(b,k)
    return newhead






