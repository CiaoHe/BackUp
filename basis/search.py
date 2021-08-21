#binary search (int)
'''
二分的本质不是单调性，是边界
二分的模板必须保证有解
'''
def bsearch_1(l,r): 
    while l<r:
        mid = l+r>>1
        if check(mid):
            #search range:[l,mid]
            r = mid
        else:
            #search range:[mid+1,r]
            l = mid+1
    return l

def bsearch_2(l,r):
    while l<r:
        mid = l+r+1>>1
        if check(mid):
            #search range: [mid,r]
            l = mid
        else:
            #search range: [l,mid-1]
            r= mid-1
    return l

'''
1 2 2 3 3 4 
Q：询问每个数的起始坐标和终止坐标
'''
def search(arr,num):
    start,end = -1,-1
    n = len(arr)
    l,r = 0,n-1
    while l<r:
        mid = (l+r)/2
        # check_start函数：num右边所有的数大于等于num
        if arr[mid]>=num: #如果check_start(mid)=True, start一定在mid/mid之前
            r = mid
        else:
            l = mid+1
    #如果num不存在arr中，结果arr[l]是第一个大于等于num的数
    if arr[l]!=num: 
        start = -1
    else:
        start = l

    l,r = 0,n-1
    while l<r:
        mid = (l+r+1)/2
        #check_end函数：num左边所有的数小于等于num
        if arr[mid]<=num:
            l = mid
        else:
            r = mid-1
    if arr[l]!=num:
        end = -1
    else:
        end = l
    return start,end

'''
float型二分求解

求平方根
'''
def sqrt(x):
    l,r = 0,max(1,x)
    while(r-l>1e-6):
        mid = (l+r)/2
        if mid*mid>=x:
            r = mid
        else:
            l = mid
    return l





