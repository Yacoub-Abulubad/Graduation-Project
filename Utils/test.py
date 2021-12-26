lst = [[1,2,3],[4,5,6],[7,8,9]]
for j in lst:
    for i in j:
        if i%2==0:
            i = 0

print(lst)