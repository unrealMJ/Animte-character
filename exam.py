nums = [1,2,3,4,5,6]
target = 15

result = {

}

min_value = 1e10

for i in range(0, len(nums) - 2):
    j = i + 1
    k = len(nums) - 1

    while j != k:
        tmp = abs(nums[j] + nums[k] + nums[i] - target)
        
        if tmp <= min_value:
            min_value = tmp
            if tmp in result:
                result[tmp].append([nums[i], nums[j], nums[k]])
            else:
                result[tmp] = [[nums[i], nums[j], nums[k]]]
        
        if tmp > target:
            k -= 1
        else:
            j += 1

print(result[min_value])
    