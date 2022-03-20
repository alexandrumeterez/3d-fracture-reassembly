import numpy as np
import sys

if __name__ == '__main__':
    arr = np.loadtxt(sys.argv[1], skiprows=1)
    np.save(sys.argv[1][:-3] + 'npy', arr)
