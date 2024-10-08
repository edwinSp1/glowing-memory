import math
import numpy as np
import itertools
import torch
import random
from icecream import ic
def has_line(mat):
    n = len(mat)
    m = len(mat[0])
    for row in range(n):
        # horizontal line
        if sum(mat[row]) / m > 0.7 and all(x > 0.5 for x in mat[row]):
            return True
    
    for col in range(m):
        # vertical line
        if sum(mat[row][col] for row in range(n)) / n > 0.7 and all(mat[row][col] > 0.5 for row in range(n)):
            return True
    
    return False

# modifies mat IN PLACE 
# and returns it
# if mat[i][j] is one then it won't do anything
# else it randomly picks it
def fill_remaining(mat):
    rows = len(mat)
    cols = len(mat[0])
    for i in range(rows):
        for j in range(cols):
            if mat[i][j] != 0:
                continue
            mat[i][j] = random.random()
    return mat 

# modifies in place
def add_line(mat, i, type="horizontal"):
    if type == 'horizontal':
        for j in range(len(mat[0])):
            mat[i][j] = min(1, random.random() + 0.6)
    else:
        for j in range(len(mat)):
            mat[j][i] = min(1, random.random() + 0.6)
    
    return mat


def create_testcases(rows, cols, num_samples):
    def create_testcase(must_have_line):
        mat = [[0 for i in range(rows)] for j in range(cols)]
        mat_has_line = 0
        if must_have_line:
            for i in range(rows):
                if random.random() < 0.3:
                    add_line(mat, i, "horizontal")
            for i in range(cols):
                if random.random() < 0.3:
                    add_line(mat, i, "vertical")
        
        fill_remaining(mat)

        # convert label to probability tensor
        label = [0, 0]
        label[has_line(mat)] = 1
        
        mat_has_line = has_line(mat)
        return (
                torch.tensor(mat, dtype=torch.float), 
                torch.tensor(label, dtype=torch.float)
        ), mat_has_line
        
    train_set = []
    amt_has_line = 0
    
    # iterate through all subsets
    for _ in range(num_samples):
        test, mat_has_line = create_testcase(must_have_line=False)
        train_set.append(test)
        amt_has_line += mat_has_line
    
    while amt_has_line < len(train_set) // 2:
        test, _ = create_testcase(must_have_line=True)
        train_set.append(test)
        amt_has_line += 1
    

    random.shuffle(train_set)
    return train_set[:num_samples]

# should be around 0.5
if __name__ == '__main__':
    test_set = create_testcases(5, 5, 100)
    ic(test_set)
    count_has_line = 0
    for mat, label in test_set:
        if label[1]:
            count_has_line += 1

    ic(count_has_line/len(test_set))

    



