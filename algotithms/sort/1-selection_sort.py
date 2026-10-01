def selection_sort(arr):
    n = len(arr)
    for i in range(n):
        max_idx = i
        for j in range(i+1, n):
            if arr[j] < arr[max_idx]:
                max_idx = j
        arr[i], arr[max_idx] = arr[max_idx], arr[i]
    return arr

def test_selection_sort():
    arr = [64, 25, 12, 22, 11]
    assert selection_sort(arr) == [11, 12, 22, 25, 64]
    print("All tests passed")
    return True

if __name__ == "__main__":
    test_selection_sort()