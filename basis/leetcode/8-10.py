#Atoi
def Atoi(s:str)->int:
    s = s.lstrip()
    if s=='': return 0

    res = 0

    sign = 1
    p = 0
    if s[p] == '-':
        sign = -1
        p+=1
    elif s[0] == '+':
        p+=1
    
    while p<len(s) and (s[p]>=str(0) and s[p]<=str(9)):
        res = res*10 + int(s[p])
        p+=1
        if res>2**31-1:
            break
    
    res*= sign
    if res>2**31-1: res=2**31-1
    elif res<-2**31: res = -2**31
    return res

#回文数

#正则表达匹配：
#非dp
def isMatch(s:str, p:str)-> bool:
    if not p: return not s

    #第一个字符是否匹配
    first_match = s and p[0] in {s[0],'.'}

    #如果p第二个字符'*':
    if len(p)>=2 and p[1]=='*':
        #星号匹配0个前面的元素: 
        '''
        s : '##'
        p : 'a*##' -> '##'
        '''
        cond1 = isMatch(s,p[2:])

        #星号匹配1个或多个前面的元素（多个通过递归逐渐降解为1个的情况）
        '''
        s: 'aaab' -> 'aab'
        p: 'a*b' 
        '''
        cond2 = first_match and isMatch(s[1:],p)
        return cond1 or cond2
    else:
        return first_match and isMatch(s[1:],p[1:])

s = "aab" 
p = "a*b"

#DP
def ismatch(s,p):
    n,m = len(s),len(p)
    s = ' '+s
    p = ' '+p
    #加' '的目的：这样指针i,j就可以取到n和m,

    dp= [[False for _ in range(m+1)] for __ in range(n+1)] #(n+1, m+1)
    dp[0][0] = True

    for i in range(0,n+1): # for(i=0; i<=n; i++)
        for j in range(1,m+1): #for(j=1; j<=m; j++) j从1开始，因为p作为一个空串不能去匹配一个非空串
            #针对a*这种情况，直接把j往后移动
            if j+1 <= m and p[j+1] == '*':
                continue
            if i>=1 and p[j]!='*':
                dp[i][j] = dp[i-1][j-1] and (s[i]==p[j] or p[j]=='.')
            elif p[j] == '*':
                dp[i][j] = dp[i][j-2] or (i and dp[i-1][j] and (s[i]==p[j-1] or p[j-1]=='.'))
    print(dp)
    return dp[n][m]

#上面那个DP写的实在是难以理解，各种边界
'''
递归：return dp[0][0]
循环: return dp[n][m]
'''
#递归
def ismatch2(s,p):
    memo = {}
    def dp(i,j):
        S,P = len(s),len(p)
        if (i,j) in memo:
            return memo[(i,j)]
        if j == P:
            #说明p已经匹配完
            return i == S
        first_match = (i<S) and (p[j] in {s[i],'.'})

        if j+1<P and p[j+1]=='*':
            tmp = dp(i,j+2) or (first_match and dp(i+1,j))
        else:
            tmp = first_match and dp(i+1,j+1)
        memo[(i,j)] = tmp
        return tmp
    return dp(0,0)

s = "a"
p = "ab*a"
print(ismatch('aa','a*'))