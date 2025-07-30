import os
import pickle
from sklearn.neighbors import KDTree
import numpy as np

def parse_mesh_header(plyfile, ext):
    # Variables
    line = []
    vertex_properties = []
    num_points = None
    num_faces = None
    current_element = None


    while b'end_header' not in line and line != b'':
        line = plyfile.readline()

        # Find point element
        if b'element vertex' in line:
            current_element = 'vertex'
            line = line.split()
            num_points = int(line[2])

        elif b'element face' in line:
            current_element = 'face'
            line = line.split()
            num_faces = int(line[2])

        elif b'property' in line:
            if current_element == 'vertex':
                line = line.split()
                vertex_properties.append((line[2].decode(), ext + ply_dtypes[line[1]]))
            elif current_element == 'vertex':
                if not line.startswith('property list uchar int'):
                    raise ValueError('Unsupported faces property : ' + line)

    return num_points, num_faces, vertex_propertiess

def parse_header(plyfile, ext):
    # Variables
    line = []
    properties = []
    num_points = None

    ply_dtypes = dict([
        (b'int8', 'i1'),
        (b'char', 'i1'),
        (b'uint8', 'u1'),
        (b'uchar', 'u1'),
        (b'int16', 'i2'),
        (b'short', 'i2'),
        (b'uint16', 'u2'),
        (b'ushort', 'u2'),
        (b'int32', 'i4'),
        (b'int', 'i4'),
        (b'uint32', 'u4'),
        (b'uint', 'u4'),
        (b'float32', 'f4'),
        (b'float', 'f4'),
        (b'float64', 'f8'),
        (b'double', 'f8')
    ])

    while b'end_header' not in line and line != b'':
        line = plyfile.readline()

        if b'element' in line:
            line = line.split()
            num_points = int(line[2])

        elif b'property' in line:
            line = line.split()
            properties.append((line[2].decode(), ext + ply_dtypes[line[1]]))

    return num_points, properties
def read_ply(filename, triangular_mesh=False):
    valid_formats = {'ascii': '', 'binary_big_endian': '>',
                 'binary_little_endian': '<'}
    with open(filename, 'rb') as plyfile:


        # Check if the file start with ply
        if b'ply' not in plyfile.readline():
            raise ValueError('The file does not start whith the word ply')

        # get binary_little/big or ascii
        fmt = plyfile.readline().split()[1].decode()
        if fmt == "ascii":
            raise ValueError('The file is not binary')

        # get extension for building the numpy dtypes
        ext = valid_formats[fmt]

        # PointCloud reader vs mesh reader
        if triangular_mesh:

            # Parse header
            num_points, num_faces, properties = parse_mesh_header(plyfile, ext)

            # Get point data
            vertex_data = np.fromfile(plyfile, dtype=properties, count=num_points)

            # Get face data
            face_properties = [('k', ext + 'u1'),
                                ('v1', ext + 'i4'),
                                ('v2', ext + 'i4'),
                                ('v3', ext + 'i4')]
            faces_data = np.fromfile(plyfile, dtype=face_properties, count=num_faces)

            # Return vertex data and concatenated faces
            faces = np.vstack((faces_data['v1'], faces_data['v2'], faces_data['v3'])).T
            data = [vertex_data, faces]

        else:

            # Parse header
            num_points, properties = parse_header(plyfile, ext)

            # Get data
            data = np.fromfile(plyfile, dtype=properties, count=num_points)

    return data
def buildKDTreeForSp(components,all_xyz):
    # 计算每个超点的中心点
    points = all_xyz
    sp_centers = []
    for superpoint in components:
        superpoint_coords = points[superpoint]  # 获取 superpoint 中所有点的坐标
        center = np.mean(superpoint_coords, axis=0)  # 计算 superpoint 的中心点坐标
        sp_centers.append(center)
    # print(sp_centers)
    # 基于中心点构建KD树
    sp_centers = np.array(sp_centers)
    tree = KDTree(sp_centers,leaf_size=15)
    # 对每个超点查询最近的个邻居的索引
    num_neighbors = 20 # 邻居数量
    neighbors_indices = tree.query(sp_centers, k=num_neighbors + 1, return_distance=False)[:, 1:]
    return tree, neighbors_indices
    # 打印每个超点的邻居索引
    # for i, indices in enumerate(neighbors_indices):
    #     print(f"超点 {i} 的最近 {num_neighbors} 个邻居的索引是: {indices}")
def process_superpoint_files(directory):
    """
    Process each .superpoint file in the specified directory.
    
    :param directory: Path to the directory containing .superpoint files
    """
    # sp_directory = os.listdir(os.path.join(directory, '0.012', 'superpoint'))
    sp_directory = directory+'/0.012/superpoint'
    for filename in os.listdir(sp_directory):
        if filename.endswith(".superpoint"):
            sp_file_path = os.path.join(directory, '0.012', 'superpoint', filename)
            cloud_name = os.path.splitext(filename)[0]
            print(f"Processing file: {cloud_name}")
            with open(sp_file_path, 'rb') as f:
                sp_obj = pickle.load(f)
            ply_file_path = os.path.join(directory, 'input_0.060', cloud_name+'.ply')
            with open(ply_file_path, 'rb') as f:
                data = read_ply(ply_file_path)
                # build KDTree
            sub_xyz = np.vstack((data['x'], data['y'], data['z'])).T
            # sub_color = np.vstack((data['red'], data['green'], data['blue'])).T
            # sub_label = data['class']
            spKDTree, sp_neighbors_indices = buildKDTreeForSp(components = sp_obj["components"], all_xyz = sub_xyz)
            sp_kd_tree_file = os.path.join(directory, '0.012','spKDTree', cloud_name + '_spKDTree.pkl')
            with open(sp_kd_tree_file, 'wb') as f:
                pickle.dump(spKDTree, f)
            sp_neighbours_file = os.path.join(directory, '0.012', 'spNeighbour', cloud_name + '_spNeighbour.pkl')
            with open(sp_neighbours_file, 'wb') as f:
                pickle.dump(sp_neighbors_indices, f)

if __name__ == "__main__":
    directory = "/mnt/home/dxy/dataset/semantic3d"
    process_superpoint_files(directory)
