1-dimension Prefix Sum

a1, a2, a3, ... an
前缀和： s_i = a_1 + ... + a_i

1.如何求s_i：
     s_0 = 0
     s_i = s_(i-1) + a_i
2.有什么用？
    可以求[l,r]段数的和
    val = s_r - s_(l_1)

N = 10000
int l, int r
int a[N]
s = [None]*N
s[0] = 0
for i in range(1,N):
    s[i] = s[i-1]+a[i]
res = s[r] - s[l-1]



###
2-dimension Prefix Sum
a_ij
s_ij

如何求s_ij?
for i in range(1,n+1):
    for j in range(1,m+1):
        s(i,j) = s(i-1,j) + s(i,j-1) - s(i-1,j-1) +a(i,j)

求左上角(x1,y1)和右下角(x2,y2)内所有元素的和：
val = s(x2,y2) - s(x2,y1-1) - s(x1-1,y2) + s(x1-1,y1-1)


###
1-d差分

a1,a2,...,an
构造b: b1,b2,...,bn; --->b是a的差分，a是b的前缀和
使得: b1=a1, b2=a2-a1, b3=a3-a2,..., bn =a_n - a_n-1

Q：有什么用？
操作： 让a[l],...,a[r]每个元素都加上c
使用差分数组b： 只需要b[l]+=c, b[r+1]-=c, 再次由b生成a时候没保证满足上述操作需求


##
2-d差分

a_ij
差分矩阵b_ij: a_ij = sum(b[1,1]+...+b[i,j])

操作需求：对a中某个子矩阵[(x1,y1),(x2,y2)]都加上一个c

b(x1,y1)+=c
b(x2+1,y1) -= c
b(x1,y2+1) -= c
b(x2+1,y2+1) += c






