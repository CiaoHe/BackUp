'''
K路归并
'''
class Solution:
    def mergeKLists(self, lists: List[ListNode]) -> ListNode:
        if not lists: return lists
        k = len(lists)
        interval = 1
        while interval<k:
            for i in range(0,k-interval,interval*2):
                lists[i] = self.merge2Lists(lists[i],lists[i+interval])
            interval*=2
        return lists[0]

    '''
    list   0    1   2   3   4   5
            \   /    \  /    \  /
             \ /      \/      \/
              0       2       4
              |       |       |
                  0           4
                  |           |
                        0
    '''
        
        
    '''
    需要熟记基本的二路归并
    '''
    def merge2Lists(self, l1, l2):
        head = prev = ListNode(-1)
        while l1 and l2:
            if l1.val<l2.val:
                prev.next = l1
                l1 = l1.next
            else:
                prev.next = l2
                l2 = l2.next
            prev = prev.next
        
        if l1:
            prev.next = l1
        if l2:
            prev.next = l2
        return head.next
# -------------------------------------------------------------------------
heap sort
'''
Max-heap: output->ascend
Min-heap: output->descend
'''
def heapSort(arr):
    n = len(arr)

    #first build heap
    for i in range((n-1)//2, -1, -1):
        createHeap(arr, n, i)
    #move root to the last, delete the last
    for i in range(n-1, -1, -1):
        arr[0], arr[i] = arr[i], arr[0]
        createHeap(arr, i, 0)

    return arr

def createHeap(arr, n, i):
    largest_index = i;
    left_index = 2*i+1
    right_index = 2*i+2

    #if left child bigger than parent, move left to parent
    if left_index<n and arr[left_index] > arr[largest_index]:
        largest_index = left_index
    #if right child bigger than parent, move right to parent   
    if right_index<n and arr[right_index] > arr[largest_index]:
        largest_index = right_index
    #if parent-node has changed
    if largest_index!= i:
        arr[largest_index], arr[i] = arr[i], arr[largest_index]
        createHeap(arr, n, largest_index)

def quickSort(arr,l,r):
    if l>=r: return arr
    #choose key
    x = arr[l]
    #left pointer and right pointer; 先移动两端指针，再进行交换
    i = l-1
    j = r+1
    while i<j:
        while True:
            i+=1
            #左边一定要<x, =都不行
            if arr[i]>=x:
                break
        while True:
            j-=1
            #右边一定要>x, =都不行
            if arr[j]<=x:
                break
        if i<j:
            arr[i],arr[j] = arr[j],arr[i]
    quickSort(arr,l,j)
    quickSort(arr,j+1,r)
    return arr

    '''
    如果用i作为边界： (1) key = arr[r] OR arr[(l+r)/2]  (2) quickSort(arr,l,i-1) quickSort(arr,i,r)
                    ！！要保证左指针i一定不能取到l这个边界

    如果用j作边界： (1) key = arr[l] (2) quickSort(arr,l,j) quickSort(arr,j+1,r)
    ?为什么用arr[l]作为key，最后的分界点是[l,j]和[j+1,r]：
    Ans: j是后于i移动的，当j停止时，证明j指向的数值是小于等于key,所以可以保证arr[l,j]都是小于等于key；同理j+1指向的数值是一定大于key的，
         可以保证arr[j+1,r]都是大于key的。

    拿sort([1,2])为例子, 如果key=arr[l]=1, 第一次停止时i=0,j=0, 则quickSort(arr,1,-1)->End quickSort(arr,0,1)陷入死循环
    '''

def mergeSort(arr,l,r):
    if l>=r: return arr
    #确定中点
    mid = (l+r)/2
    #递归mergeSort
    mergeSort(arr,l,mid)
    mergeSort(arr,mid+1,r)
    #归并
    tmp=[]
    #two pointers
    i = l #left part start
    j = mid+1 #right part start
    while i<=mid and j<=r:
        if q[i]<=q[j]:
            tmp.append(q[i])
            i+=1
        else:
            tmp.append(q[j])
            j+=1
    while i<=mid:
        tmp.append(q[i])
        i+=1
    while j<=r:
        tmp.append(q[j])
        j+=1
    arr = tmp
    return arr

 #k-th min
int quickSort(int l, int r, int k):
    if (l==r): return q[l]
    x = q[l], i =l-1, j = r+1
    while(i<j):
        while True:
            i+=1
            if q[i]>=x:
                break
        while True:
            j-=1
            if q[j]<=x:
                break
        if i<j:
            q[i],q[j] = q[j],q[i]
    #left part:[l,j], right part:[j+1,r]
    sl = j-l+1
    if k<=sl:
        return quickSort(l,j,k)
    return quickSort(j+1,r,k-sl)


#逆序对数目

l = 0
r = len(q)-1
print(f'ans: {mergeSort(l,r)}')

def mergeSort(l,r)->int:num:
    #return num of inverse pair
    if l>=r: return 0
    mid = l+r>>1
    num = mergeSort(l,mid) + mergeSort(mid+1,r)

    tmp = []
    i = l, j = mid+1
    while i<=mid and j<=r:
        if q[i]<=q[j]:
            tmp.append(q[i])
            i+=1
        else:
            res+=mid-i+1 #加这一行来计数黄色情况的逆序对数目
            tmp.append(q[j])
            j+=1
    while i<=mid:
        tmp.append(q[i])
        i+=1
    while j<=r:
        tmp.append(q[j])
        j+=1
    q = tmp
    return res






