# HP-CAL: Active Learning with Holistic Perception and Category Awareness for Point Cloud Semantic Segmentation

 ## Environment
 1. Setup python environment.
```
conda create -n ssdr python=3.8
source activate ssdr
pip install -r helper_requirements.txt
sh compile_op.sh
```
2. Install additional Python packages.
```
pip install future python-igraph tqdm transforms3d pynvrtc fastrlock cupy h5py sklearn plyfile scipy pandas
```
3.Install Boost and Eigen3, in Conda.
```
conda install -c anaconda boost; conda install -c omnia eigen3; conda install eigen; conda install -c r libiconv
```
4. Compile the libply_C and libcp libraries.
```
CONDAENV=YOUR_CONDA_ENVIRONMENT_LOCATION
cd partition/ply_c
cmake . -DPYTHON_LIBRARY=$CONDAENV/lib/libpython3.6m.so -DPYTHON_INCLUDE_DIR=$CONDAENV/include/python3.6m -DBOOST_INCLUDEDIR=$CONDAENV/include -DEIGEN3_INCLUDE_DIR=$CONDAENV/include/eigen3
make
cd ..
cd cut-pursuit
mkdir build
cd build
cmake .. -DPYTHON_LIBRARY=$CONDAENV/lib/libpython3.6m.so -DPYTHON_INCLUDE_DIR=$CONDAENV/include/python3.6m -DBOOST_INCLUDEDIR=$CONDAENV/include -DEIGEN3_INCLUDE_DIR=$CONDAENV/include/eigen3
make
```

## Dataset
- ## S3DIS 

Download the files named "Stanford3dDataset_v1.2_Aligned_Version.zip". Uncompress the folder and move it to /dataset/S3DIS.
1. preparing the dataset.
```
python utils/data_prepare_s3dis.py
```
2. Compute the superpoint.
```
python partition/compute_superpoint_s3dis.py
```
3. Compute the neighborhood superpoint.
```
python utils/buildKDTree4s3dis.py
```
- ## Semantic3D

Download and extract the dataset. First, please specify the path of the dataset by changing the BASE_DIR in "download_semantic3d.sh"
 ```
sh utils/download_semantic3d.sh
```
1. preparing the dataset.
```
python utils/data_prepare_semantic3d.py
```
2. Compute the superpoint.
```
python partition/compute_superpoint_semantic3d.py
```
3. Compute the neighborhood superpoint.
```
python utils/buildKDTree4semantic3d.py
```
## Run

- ## S3DIS 
```
python ssdr_main_S3DIS2.py
```

- ## Semantic3D

```
python ssdr_main_semantic3d.py
```
