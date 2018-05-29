

class BinaryTreeSort():
    def __init__(self, data):
        self.left = None
        self.right = None
        self.data = data

    def insert(self, data):
        if len(data) <= len(self.data):
            if self.left is None:
                self.left = BinaryTreeSort(data)
            else:
                self.left.insert(data)
        else:
            if self.right is None:
                self.right = BinaryTreeSort(data)
            else:
                self.right.insert(data)

    def print_tree(self, s):
        if self.left is not None:
            self.left.print_tree(s)
        # print(self.data)
        s.append(self.data)
        if self.right is not None:
            self.right.print_tree(s)



# def partition(arr, l, h):
#     i = (l - 1)
#     x = arr[h]
#
#     for j in range(l, h):
#         if arr[j] <= x:
#             # increment index of smaller element
#             i = i + 1
#             arr[i], arr[j] = arr[j], arr[i]
#
#     arr[i + 1], arr[h] = arr[h], arr[i + 1]
#     return (i + 1)


def quick_sort_iterative(list_, left, right, y):
    """
    Iterative version of quick sort
    """
    temp_stack = []
    temp_stack.append((left, right))

    # Main loop to pop and push items until stack is empty
    while temp_stack:
        pos = temp_stack.pop()
        right, left = pos[1], pos[0]
        piv = partition(list_, left, right, y)
        # If items in the left of the pivot push them to the stack
        if piv - 1 > left:
            temp_stack.append((left, piv - 1))
        # If items in the right of the pivot push them to the stack
        if piv + 1 < right:
            temp_stack.append((piv + 1, right))


def quick_sort_recursive(list_, left, right):
    """
    Quick sort method (Recursive)
    """
    if right <= left:
        return
    else:
        # Get pivot
        piv = partition(list_, left, right)
        # Sort left side of pivot
        quick_sort(list_, left, piv - 1)
        # Sort right side of pivot
        quick_sort(list_, piv + 1, right)


def partition(list_, left, right, y):
    """
    Partition method
    """
    # Pivot first element in the array
    piv = len(list_[left])
    i = left + 1
    j = right

    while 1:
        while i <= j and len(list_[i]) <= piv:
            i += 1
        while j >= i and len(list_[j]) >= piv:
            j -= 1
        if j <= i:
            break
        # Exchange items
        list_[i], list_[j] = list_[j], list_[i]
        y[i], y[j] = y[j], y[i]
    # Exchange pivot to the right position
    list_[left], list_[j] = list_[j], list_[left]
    y[left], y[j] = y[j], y[left]
    return j