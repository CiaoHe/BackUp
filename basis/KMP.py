dp[j][c] = next
'''
0 <= j < m: 代表当前状态
c:[0,256): 代表当前遇到的字符
0 <= next <= m: 代表下一个状态å

状态数目总数：m+1 (最后一个代表完成状态)
'''
def search(txt):
    m = len(pat)
    n = len(txt)
    j = 0
    i = 0
    while i<n:
        #当前是状态j,遇到了字符txt[i]，下一个状态应该是：
        j = dp[j][txt[i]]
        #如果达到了终止状态，返回匹配的开头索引
        if j==m:
            return i-m+1
        i+=1
############################################################
#大问题：如何构建dp?

x: int #影子状态
for j in range(m):
    for c in range(256):
        #如果遇到的c符合j状态下伸的桥梁,推进j状态带j+1
        if c == pat[j]:
            dp[j][c] = j+1
        else:
            #状态重启
            #委托 X 计算重启位置
            dp[j][c] = dp[x][c] 

############################################################
#最后一个问题： 如何求出影子状态X？
dp = [[0 for _n in range(256)] for _m in range(m)] #!死亡教训：永远不要偷懒写成[0]*m
#base case
#从状态0,只有匹配了pat[0]才会跳转到下一个状态1
dp[0][pat[0]] = 1

x = 0
for j in range(1,m):
    #对于所有可能遇到的字符c，我们如何跳转呢？
    for c in range(256):
        #遇到的字符c正好为pat[j]，跳到下一个状态
        if c == pat[j]:
            dp[j][c] = j+1
        #否则，聪明的影子x会告诉我们：
        else:
            dp[j][c] = dp[x][c]
    #update x
    x = dp[x][pat[j]]

#综上：
def KMP(pat):
    m = len(pat)
    dp = [[0 for _n in range(256)] for _m in range(m)]
    dp[0][orf(pat[0])] = 1

    x = 0
    for j in range(1,m):
        for c in range(256):
            if c == ord(pat[j]):
                dp[j][c] = j+1
            else:
                dp[j][c] = dp[x][c]
        x = dp[x][pat[j]]
    return dp


'''
Q28
'''
def KMP(pat):
    m = len(pat)
    dp = [[0 for _n in range(256)] for _m in range(m)]
    dp[0][ord(pat[0])] = 1

    x = 0
    for j in range(1,m):
        for c in range(256):
            dp[j][c] = int(dp[x][c])
        dp[j][ord(pat[j])] = j+1
        x = dp[x][ord(pat[j])]
    return dp

def search(pat,txt):
    dp = KMP(pat)
    m,n = len(pat),len(txt)
    j:int = 0
    for i in range(n):
        j = dp[j][ord(txt[i])]
        if j == m:
            return i-m+1
    return -1

pat = 'll'
txt = 'hello'
print(search(pat,txt))



