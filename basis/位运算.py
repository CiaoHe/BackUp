n>>k &1 #看n的第k位是几

def lowbit(x):
    #返回x的最后一位1
    return x & -x # x&(~x+1)

def countOne(x):
    '''
    统计x内1的数量
    '''
    while (x!=0):
        x -= lowbit(x)
        cnt++
    return cnt

