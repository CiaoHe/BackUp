#17. 
def letterCombinations(digits: str):
    if not digits: return []
    phone = {'2':['a','b','c'],
                 '3':['d','e','f'],
                 '4':['g','h','i'],
                 '5':['j','k','l'],
                 '6':['m','n','o'],
                 '7':['p','q','r','s'],
                 '8':['t','u','v'],
                 '9':['w','x','y','z']}
    res = []
    def backtrack(combination, nextdigit):
        if len(nextdigit) == 0:
            res.append(combination)
        else:
            for char in phone[nextdigit[0]]:
                backtrack(combination+char, nextdigit[1:])
    backtrack('',digits)
    return res

#使用队列
def letterCombinations2(digits):
    queue = ['']
    phone = {'2':['a','b','c'],
                 '3':['d','e','f'],
                 '4':['g','h','i'],
                 '5':['j','k','l'],
                 '6':['m','n','o'],
                 '7':['p','q','r','s'],
                 '8':['t','u','v'],
                 '9':['w','x','y','z']}
    for digit in digits:
        for _ in range(len(queue)):
            tmp = queue.pop(0)
            for letter in phone[digit]:
                queue.append(tmp+letter)
    return queue

#18 四数的和
def fourSum(nums, target):
    res = []
    nums = sorted(nums)
    n = len(nums)
    for i in range(n-3):
        if i and nums[i-1] == nums[i]:
            continue
        #如果我现在跟最后三个在一起都小于target,我这个i不行
        if nums[i]+nums[n-3]+nums[n-2]+nums[n-1]<target:
            continue
        #如果i,i+1,i+2,i+3加在一起都大于target，提前stop
        if nums[i]+nums[i+1]+nums[i+2]+nums[i+3]>target:
            break

        #进入第二重循环
        for j in range(i+1,n-2):
            if j>i+1 and nums[j-1]==nums[j]:
                continue
            if nums[i]+nums[j]+nums[n-2]+nums[n-1]<target:
                continue
            if nums[i]+nums[j]+nums[j+1]+nums[j+2]>target:
                break

            #进入第三重循环，上双指针
            l,r = j+1, n-1
            while l<r:
                if l>j+1 and nums[l]==nums[l-1]:
                    l+=1
                    continue
                while l<r-1 and nums[i]+nums[j]+nums[l]+nums[r-1]>=target:
                    r-=1
                if nums[i]+nums[j]+nums[l]+nums[r]==target:
                    res.append([nums[i],nums[j],nums[l],nums[r]])
                # while l<r and nums[i]+nums[j]+nums[l]+nums[r]>target:
                #     r-=1
                # if nums[i]+nums[j]+nums[l]+nums[r] == target and l<r:
                #     res.append(...)
                l+=1
    return res

#19 
    def removeNthFromEnd(self, head: ListNode, n: int) -> ListNode:
        dummy = ListNode(-1,head)
        fir = dummy
        sec = dummy
        for _ in range(n):
            fir = fir.next
        while fir.next:
            sec = sec.next
            fir = fir.next
        sec.next = sec.next.next
        return dummy.next