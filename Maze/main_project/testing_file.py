import pandas as pd
import numpy as np

# we learned shape, ndim, and itemsize, as well as defining via datatype
# then we learned indexing with colons. [start, stop, step for specificity]
# then we learned initialization with functions like random, zeros, one, full, full_sample NOTE: use copy() function to copy b/c shared mem
# for arithmetic, you can do entire functions on the np array. There are also built in math functions like sin. Targeting also works
# statistics are performed using functions such as sum, max, and min.
# mixing arrays with hstack and vstack
# boolean masking and advanced indexing
# any? do any values in the axis meet the parameters

def main():
    a = np.linspace(1,30, 30).reshape(6,5)
    print(a[2:-2, 0:2])
    b = np.identity(4)
    c = a[[0,1,2,3],[1,2,3,4]]
    d = a[[1,-2,-1], 3:]
    print(d)
    

if __name__ == "__main__":
    main()