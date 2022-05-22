#%%
import matplotlib
import numpy as np
import os
import matplotlib.pyplot as plt


def read_pcd(file):
    with open(file) as f:
        lines = f.readlines()
        point_cloud = np.array([x.split(" ") for x in lines[1:]], dtype=np.float32)
    return point_cloud

files = os.listdir(".")
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
v = 0
for file in files:
    if file[-1] != "d":
        continue
    point_cloud = read_pcd(file)
    v += point_cloud.shape[0]
    ax.scatter(point_cloud[:,0],point_cloud[:,1],point_cloud[:,2],s=1, marker = 'o')
plt.show()

# %%
