"""
A+B
A-B
A*b
A/b: 求商,余数
"""
""""""""""""""""""""""""""""""""""""""""""""""""""
#Add
N= 1e6+10
a = '1213111...'
b = '23322423...'
#input str-> int
A,B =[],[]
for i in range(len(a)-1,-1,-1):
    A.append(int(a[i]))
for j in range(len(b)-1,-1,-1):
    B.append(int(b[j]))
#per bit add
def add(A,B):
    C = []
    t = 0#进位
    for i in range(max(len(A),len(B))):
        if i<len(A):
            t+=A[i]
        if i<len(B):
            t+=B[i]
        C.append(t%10)
        t  = int(t/10)
    if t:
        C.append(1)
    return C[::-1]
""""""""""""""""""""""""""""""""""""""""""""""""""

#Sub
A,B =[],[]
for i in range(len(a)-1,-1,-1):
    A.append(int(a[i]))
for j in range(len(b)-1,-1,-1):
    B.append(int(b[j]))
#judge which one is bigger
def cmp(A,B):
    if len(A)!=len(B): 
        return len(A)>len(B)
    else:
        for i in range(len(A)):
            if A[i]!=B[i]:
                return A[i]>B[i]
        return True

#per bit sub
def sub(A,B):
    C = []
    t = 0
    for i in range(len(A)):
        t = A[i] - t
        if i<len(B):
            t -= B[i]
        C.append((t+10)%10)
        if t<0:
            t = 1
        else:
            t = 0
    #remove all 0 in invalid bit (e.g. 300->3)
    while len(C)>1 and C[-1]==0:
        C = C[:-1]
    return C[::-1]

if cmp(A,B):
    return sub(A,B)
else:
    return '-'+sub(B,A)

################################################
#Multiply
a = '1213234343'
b = 12
A = [a[i] for i in range(len(A)-1,-1,-1)]

def mul(A,b):
    C = []
    t = 0
    i = 0
    while i<len(A) or t:
        if i<len(A):
            t+=A[i]*b
        C.append(t%10)
        t = int(t/10)
        i+=1
    return C[::-1]

################################################
#Divide
a = '1213234343'
b = 12
A = [a[i] for i in range(len(a))]
def divide(A,b):
    C = []
    r = 0
    i = 0
    while i<len(A):
        r = r*10+A[i]
        if int(r/b):
            C.append(int(r/b))
        else:
            C.append(0)
        r = int(r%b)
        i+=1
    while len(C)>1 and C[0]==0:
        C = C[1:]
    return C






