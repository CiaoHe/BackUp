def func(s):
    d = {} #记录各个char出现的次数，实际上要维持都是1
    i,j = 0,0
    res = 0
    while(i<len(s)):
        #给i遇上的char计数器先统统+1
        if s[i] not in d:
            d[s[i]] = 1
        else:
            d[s[i]]+=1

        #i指向的char出现多余1次，开始用j清场直到保证counter(s[i])=1，
        while (d[s[i]]>1):
            d[s[j]]-=1
            j+=1
        res = max(res, i-j+1)
        i+=1
    return res

#d = collections.defaultdict(0)

def median(nums1,nums2):
    m,n = len(nums1),len(nums2)
    tot = m+n
    if (tot%2 == 0):
        # tot是偶数，偶数个数字的中位数=(left+right)/2
        left = find(nums1, 0, nums2, 0, tot//2) #find(nums1, nums1的起点，nums2, nums2的起点，要找到第tot/2个大的数)
        right = find(nums1, 0, nums2, 0, tot//2+1) #find(nums1, nums1的起点，nums2, nums2的起点，要找到第（tot/2）+1个大的数)
        return (left+right)/2
    else:
        return find(nums1, 0, nums2, 0, tot//2+1)

def find(nums1, i, nums2, j, k):
    #find the kth number in nums1 & nums2
    #!!!!注意这里的i,j,k的index类型不同, i&j是从0开始计数，k是从1开始计数
    #>>>>我们定义int有两种type，1）从0开始计数的index type 2）从1开始计数的counter type
    #>>>>转换二者，counter-1 --> index

    #For convinience, Aussume nums1 shorter
    if len(nums1)-i > len(nums2) - j:
        return find(nums2, j, nums1, i, k)
    
    #处理边界
    #1. 如果k=1
    if k==1:
        if len(nums1) == i: return nums2[j]  #len(nums1) == i 这是一种表空的方法(i作为nums1的指针)
        else: return min(nums1[i], nums2[j])
    #2. 如果nums1==[]:
    if len(nums1) == i:
        return nums2[j+k-1]
    
    #开始正式讨论
    si = min(i + k//2 - 1, len(nums1)-1) #对于nums1的第k/2个数, 注意要取min,防止越界，因为当nums1只有一个元素的时候,si很容易成为1
    sj = j + (k-k//2) - 1 #对于nums2的第k/2个数（因为有奇偶问题，所以统一表示为(k-k/2)

    if nums1[si] > nums2[sj]:
        return self.find(nums1, i, nums2, sj+1, k-(sj-j+1))
    else:
        return self.find(nums1, si+1, nums2, j, k-(si-i+1))

print(median([1],[2,3,4,5,6]))