for i in range(n):
    j = 0
    while j<i and condition:
        j++

# double pointers O(2n)

s = 'abc def ghi'
#output each word in each line
i=0
while i<len(s):
    j=i
    while j<len(s) and s[j]!=' ':
        j+=1
    res.append(s[i:j])
    i = j+1

#最长连续不重复子数列
i,j=0,0
while i < len(s):
    while j<=i and checkDup(i,j):
        j+=1
    res = max(res,i-j+1)
    i+=1

def checkDup(i,j):
    #i:right-end j:left-end
    return len(s[j:i]) != len(set(s[j:i]))

s[N]
a[N]
for (i=0,j=0;i<n;i++)
{
    s[a[i]]++;
    while (s[a[i]]>1) :
    {
        s[a[j]]--;
        j++;
    }
    res = max(res,i-j+1)
}