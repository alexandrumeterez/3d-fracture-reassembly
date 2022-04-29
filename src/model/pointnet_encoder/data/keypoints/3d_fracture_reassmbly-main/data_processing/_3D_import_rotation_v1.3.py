import pandas as pd
import numpy as np
from scipy.spatial.transform import Rotation
from scipy.spatial.distance import pdist
import time

start=time.time()
print('Importing data...')

#---------------------------------------------------Settings-----------------------------------------------------------------------
# name of the file to open
filename='fractured_cuboid.obj'
# name of the file to generate
out_filename='fractured_cuboid_rot.obj'
# seed of the random rotation and translation generator set to integer for reproducible results, set to None for random result
randseed=36
# set to true, if a .obj file should be generated from the rotated fragments
generate_output=True
# standard deviation for random translation. Should be ~ 0.5 * max distance. To infer from part, set to 0 (WARNING: slow for many vertices)
sigma=0
# --------------------------------------------------------------------------------------------------------------------------------

# find the longest row in the input file by iterating through the rows
col_count=[]
with open(filename, 'r') as temp_f:
    for l in temp_f.readlines():
        if l[0]!='#':   # ignore commented lines
            col_count.append(len(l.split(" "))) # number of columns per row
#create column names
column_names = [i for i in range(0, max(col_count))]

#open the file, create data frame
df = pd.read_csv(filename, delim_whitespace=True, header=None, index_col=0, names=column_names, comment='#', dtype='str')

# search for the seperate objects in the file (indicated by 'o')
fragmentindex=[]
for index, is_object in enumerate(df.index.get_loc('o')): # iterate through boolean vector: True, if row begins with 'o'
    if is_object==True:
        fragmentindex.append(index) # index of the rows, where a new object begins
fragmentindex.append(index)         # add the last index of the file (for easier slicing)

# in the rows with the faces, extract integer before the first '/' (= vertices contained in that face)
face_vertex=pd.concat([df.loc[:,i].str.split('/', expand=True)[0].rename(i) for i in range(1, max(col_count))], axis=1)

# slice the input dataframe in the seperate objects, store the fragments in seperate dataframes
fragments=[]
for i in range(len(fragmentindex)-1):
    fragments.append(face_vertex.iloc[fragmentindex[i]:fragmentindex[i+1]])

# get the vertices, surface normals and face vertex indices of every fragment as numpy array
# faces have varying number of vertices, empty places are represented as negative numbers
vertices=[]
normals=[]
faces=[]
for fragment in fragments:
    vertices.append(fragment.loc['v',1:3].to_numpy(dtype='float'))
    normals.append(fragment.loc['vn',1:3].to_numpy(dtype='float'))
    faces.append(fragment.loc['f'].fillna(value=-1).to_numpy(dtype='int'))

# get indices of the vertices and surface normals contained in each surface
f_vertices=[]
vert_offset=0
norm_offset=0
for i in range(len(fragments)):
    # subtract offset, to get indices of vertices[i]
    f_vertices.append(faces[i]-vert_offset-1)
    # convert to list, delete negative entries
    f_vertices[i]=f_vertices[i].tolist()
    for j in range(len(f_vertices[i])):
        for k in range (len(f_vertices[i][j])):
            if f_vertices[i][j][k] < 0:
                del f_vertices[i][j][k:]
                break
    
    vert_offset+=vertices[i].shape[0]
    norm_offset+=normals[i].shape[0]

print('Imported '+str(vert_offset)+' vertices and '+str(norm_offset)+' faces from '+str(i+1)+' fragments')
print('Done! ('+str(round(time.time()-start,4))+' seconds)')
start=time.time()
print('Rotating and translating fragments ...')

# f_vertices[fragment][# of face] = indices of associated vertices as list of integers
# vertices[fragment] = N x 3 numpy array of vertices (row: index of vertex, column: x, y, z cartesian coordinates)
# normals[fragment] = M x 3 numpy array of surface normals (row: index of surface normal (=index of face), x, y, z cartesian coordinates of unit vector in normal direction)

# initialize variables
quat=np.zeros([4])
rot=[]
trans=[]
vertices_new=[]
normals_new=[]
# initialize random generator
rng=np.random.default_rng(seed=randseed)

# calculate max distance between vertices, set sigma=max/2 (standard deviation): 95,45% of all translation vectors are within the max distance
if sigma == 0: sigma=np.amax(pdist(np.array(df.loc['v',0:3].astype('float')),'euclidean'))/2.
print ('Used standard deviation: sigma = '+str(sigma))

for i in range(len(fragments)):

    # random rotation angle in range [-pi/2, pi/2) (uniform dist.)
    theta=rng.uniform(-np.pi/2,np.pi/2)

    # random rotation direction (normal vector (uniformly distributed direction), see: https://towardsdatascience.com/the-best-way-to-pick-a-unit-vector-7bd0cc54f9b)
    n=rng.standard_normal(3)
    n=n/np.linalg.norm(n)

    # use quarternions for defining the rotation (advantage: has no singularity)
    quat[0:3]=n*np.sin(theta/2)
    quat[3]=np.cos(theta/2)

    # r is a Scipy Rotation object, see: https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.Rotation.html#scipy.spatial.transform.Rotation
    r=Rotation.from_quat(quat)
    
    # create random translation vector (normal distribution, zero mean)
    t=rng.normal(0,sigma,3)
    
    # save vector + rotation 
    rot.append(r)
    trans.append(t)

    # apply it to vertices and normals
    vertices_new.append(rot[i].apply(vertices[i])+trans[i])
    normals_new.append(rot[i].apply(normals[i]))

print('Done! ('+str(round(time.time()-start,4))+' seconds)')

# vertices_new: vertices of the fragments, every fragment randomly rotated and translated
# normals_new: surface normals of the faces, rotated equally as the vertices
# f_vertices stays the same, since its only a collection of indices (indices of vertices and normals not affected by rotation)

# rot[fragments]: Scipy Rotation object of the corresponding fragment
# trans[fragment]: 3D translation vector of the corresponding fragment

# write rotated objects to .obj file (texture coordinates 'vt' stay unchanged)
if generate_output:

    start=time.time()
    print('Writing the rotated and translated fragments to .obj file...')
    # concatenate all new vertices and normals to one numpy array
    all_vert=np.concatenate([vertices_new[i] for i in range(len(fragments))])
    all_norm=np.concatenate([normals_new[i] for i in range(len(fragments))])

    # initialize output DataFrame to input DataFrame, add comment to the first line
    comment=np.array([['rotated', 'file', 'generated', 'from', filename]])
    df_out=pd.DataFrame(comment, index=['#'], columns=range(1, comment.shape[1]+1))
    df_out=df_out.append(df)

    # replace vertices and normals with new ones
    df_out.loc['v',1:3]=all_vert
    df_out.loc['vn',1:3]=all_norm
    
    # write the DataFrame to file
    df_out.to_csv(out_filename, header=False, sep=' ')
    print('Done! ('+str(round(time.time()-start,4))+' seconds)')