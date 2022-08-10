import random
import math

num_arr = []

random.seed(2)

for x in range(4072):
    num_arr.append(random.randint(0, 64))

print(num_arr)

bool_arr = []

for x in range(len(num_arr)):
    bool_arr.append(num_arr[x] < 16)

index_arr = [int(x) for x in bool_arr]
index_arr_new = [int(x) for x in bool_arr]

print(0)
print(index_arr)

for x in range(1,int(math.log(len(num_arr), 2))+2):
    for i in range(len(num_arr)):
        if i%2**x < (2**x)/2:
            continue
        else:
            index_arr_new[i] += index_arr[i-(i%2**x)+(2**(x-1)-1)]
            
    print(2**x)
    print(index_arr_new)
    index_arr = index_arr_new
