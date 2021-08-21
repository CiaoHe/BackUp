#11. 盛最多水的容器
'''
双指针i,j，一头一尾，关键看怎么移动
'''
def maxArea(hs):
    i,j = 0,len(hs)-1
    res = 0
    while i<j:
        area = (j-i)*min(hs[i],hs[j])
        res = max(area,res)
        if hs[i]<hs[j]:
            i+=1
        else:
            j-=1
    return res



#12.整数转罗马数字
def intToRoman(num):
    reps = ['M','CM','D','CD','C','XC','L','XL','X','IX','V','IV','I']
    values=[1000, 900, 500, 400, 100, 90, 50, 40, 10, 9, 5, 4, 1]

    res = ''
    for i in range(len(reps)):
        while(num>=values[i]):
            res += reps[i]
            num -= values[i]
    
    return res

#13. 罗马数字转整数
def romanToInt(s):
    reps = ['M','CM','D','CD','C','XC','L','XL','X','IX','V','IV','I']
    values=[1000, 900, 500, 400, 100, 90, 50, 40, 10, 9, 5, 4, 1]
    d = dict(zip(reps,values))
    i = 0
    res = 0
    while i<len(s):
        if i+1<len(s) and s[i:i+2] in d:
            res+=d[s[i:i+2]]
            i+=2
        else:
            res+=d[s[i]]
            i+=1
    return res

#14 最长公共前缀
def longestCommonPrefix(strs):
    if not strs:
        return ''
    res = ''
    std = strs[0]
    for index,c in enumerate(std):
        for other in strs[1:]:
            if index<len(other) and other[index]==c:
                continue
            else:
                return res
        res += c
    return res

#15 Three Sum
def threeSum(nums):
    res = []
    nums = sorted(nums)
    for i in range(len(nums)):
        #靠小端去重
        if i>0 and nums[i] == nums[i-1]:
            continue
        #如果我都大于0了，就可以放弃了
        if nums[i]>0:
            break
        tar = nums[i]
        l,r = i+1, len(nums)-1
        while(l<r):
            #双指针动起来
            #首先，如果l这个位置使得结果不够凑齐，l向右边移动
            if tar+nums[l]+nums[r] < 0:
                l+=1
            #r这个位置使得结果大了，r向左边移动
            elif tar+nums[l]+nums[r] > 0:
                r-=1
            #正好凑齐
            else:
                res.append([tar,nums[l],nums[r]])
                l+=1
                r-=1
                #开始去重
                while(l<r and nums[l]==nums[l-1]):
                    l+=1
                while(l<r and nums[r]==nums[r+1]):
                    r-=1
    return res

def threeSum2(nums):
    res = []
    nums = sorted(nums)
    for i in range(len(nums)):
        if i and nums[i-1] == nums[i]:
            continue
        j = i+1
        k = len(nums)-1
        while j<k:
            if j>i+1 and nums[j-1] == nums[j]:
                j+=1
                continue
            #精髓！！ 用k-1和>=0, 如果不成立那么说明nums[i]+nums[j]+nums[k-1]必须小于0，
            # 至少k这个位置一旦满足成立就肯定是最靠近的一个k(最小的k)， 不需要去重k了
            while j<k-1 and nums[i]+nums[j]+nums[k-1]>=0:
                k-=1
            if nums[i]+nums[j]+nums[k]==0:
                res.append([nums[i],nums[j],nums[k]])
            j+=1
    return res

#16. 最接近的三个数之和：
'''
##不用去重
nums[i] + nums[j] +nums[k] >= tar: 这个k就是大于等于tar成立的最小k
nums[i] + nums[j] +nums[k-1] < tar: k-1一定数小于tar成立的位置
'''
def threeSumClosest(nums, target):
    res = 10e5
    distance = 10e5
    nums = sorted(nums)
    for i in range(len(nums)):
        j = i+1
        k = len(nums)-1
        while j<k:
            while j<k-1 and nums[i]+nums[j]+nums[k-1]-target>=0:
                k-=1
            if abs(nums[i]+nums[j]+nums[k]-target) < distance:
                distance = abs(nums[i]+nums[j]+nums[k]-target)
                res = nums[i]+nums[j]+nums[k]
            if j<k-1 and abs(nums[i]+nums[j]+nums[k-1]-target) < distance:
                distance = abs(nums[i]+nums[j]+nums[k-1]-target)
                res = nums[i]+nums[j]+nums[k-1]
            j += 1
    return res

