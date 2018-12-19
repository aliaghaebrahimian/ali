n = int(input())

a=[int(x) for x in input().split()]
if 2<=n<=10:
    b=list(set(a))
    b.remove(max(tmp))
print(max(b))
