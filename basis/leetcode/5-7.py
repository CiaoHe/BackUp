# 最长回文子串
def longest(s):
    res = ''
    #枚举中心点
    i = 0
    while i < len(s):
        #奇数长度
        l = i-1
        r = i+1
        while l>=0 and r<len(s) and s[l]==s[r]:
            l-=1
            r+=1
        if r-l-1 > len(res):
            res = s[l+1:r]

        #偶数长度
        l = i
        r = i+1
        while l>=0 and r<len(s) and s[l]==s[r]:
            l-=1
            r+=1
        if r-l-1 > len(res):
            res = s[l+1:r]
        
        i+=1
    return res


#Z字形转换
s = "LEETCODEISHIRING"
n = 4
def covert(s,n):
    #边界特例： s='A‘, n=1 
    if n==1: return s

    res = ''
    for i in range(n):
        if i == 0 or i==n-1:
            j = i
            while j<len(s):
                res += s[j]
                j += 2*n-2
        else:
            j = i
            k = 2*n-2-i
            while j<len(s) or k<len(s):
                res+=s[j]
                if k<len(s):
                    res += s[k]
                j += 2*n - 2
                k += 2*n - 2
    return res

#输出1到n*m的蛇形矩阵 (n,m)
def snake(n,m):
    left,right,top,bottom = 0,m-1,0,n-1
    res = [[0 for i in range(m)] for j in range(n)]

    k = 1 #填充数
    while(left<=right and top<=bottom):
        for i in range(left,right+1):
            res[top][i] = k
            k+=1
        for i in range(top+1,bottom+1):
            res[i][right] = k
            k+=1
        for i in range(right-1,left-1,-1):
            res[bottom][i] = k
            k+=1
        for i in range(bottom-1,top,-1):
            res[i][left] = k
            k+=1
        left+=1
        right-=1
        top+=1
        bottom-=1
    return res

#整数反转
def reverse(x:int) -> int:
    if x<0:
        x = '-'+str(x)[1:][::-1]
    else:
        x = str(x)[::-1]
    if x>2**31-1 or x<-2**31:
        return 0
    return int(x)


