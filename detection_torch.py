import os
import open3d as o3d
import open3d.ml as _ml3d
import open3d.ml.torch as ml3d

from open3d.ml.vis import Visualizer, BoundingBox3D, LabelLUT
from tqdm import tqdm
import time

import numpy as np

import glob

def prepare_point_cloud_for_inference(pcd):
	# Remove NaNs and infinity values
	pcd.remove_non_finite_points()
	# Extract the xyz points
	xyz = pcd.points

	# PointPillars classifier needs a 4th dimension (intensity), which my custom data does not have.
	# We add it here with default value of 0.5
	xyzi = []
	for point in xyz:
		xyzi.append(list(point) + [0.5])
	xyzi = np.array(xyzi)
	
	# Set the points to the correct format for inference
	data = {"point":xyzi, 'feat': None, 'label':np.zeros((len(xyz),), dtype=np.int32)}

	return data, pcd

def load_custom_dataset(dataset_path, candidates_number = 1000, step = 1):
	print("Loading custom dataset")
	pcd_paths = glob.glob(dataset_path+"/*.pcd")
	pcds = []
	for count, pcd_path in enumerate(pcd_paths):
		if count % step == 0:
			pcds.append(o3d.io.read_point_cloud(pcd_path))

		if count == candidates_number:
			break

	return pcds


def filter_detections(detections, min_conf = 0.5):
	good_detections = []
	for detection in detections:
		if detection.confidence >= min_conf:
			good_detections.append(detection)
	return good_detections

# Load an ML configuration file
cfg_file = "/home/carlos/Open3D/build/Open3D-ML/ml3d/configs/pointpillars_kitti.yml"
cfg = _ml3d.utils.Config.load_from_file(cfg_file)

# Load the PointPillars model
model = ml3d.models.PointPillars(**cfg.model)

# Add path to the Kitti dataset and your own custom dataset
cfg.dataset['dataset_path'] = '/media/carlos/SeagateExpansionDrive/kitti/Kitti'
cfg.dataset['custom_dataset_path'] = './pcds'

# Load the datasets
dataset = ml3d.datasets.KITTI(cfg.dataset.pop('dataset_path', None), **cfg.dataset)
custom_dataset = load_custom_dataset(cfg.dataset.pop('custom_dataset_path', None))

# Create the ML pipeline
pipeline = ml3d.pipelines.ObjectDetection(model, dataset=dataset, device="gpu", **cfg.pipeline)

# download the weights.
ckpt_folder = "./logs/"
os.makedirs(ckpt_folder, exist_ok=True)

ckpt_path = ckpt_folder + "pointpillars_kitti_202012221652utc.pth"
pointpillar_url = "https://storage.googleapis.com/open3d-releases/model-zoo/pointpillars_kitti_202012221652utc.pth"

if not os.path.exists(ckpt_path):
	cmd = "wget {} -O {}".format(pointpillar_url, ckpt_path)
	os.system(cmd)


# load the parameters of the model
pipeline.load_ckpt(ckpt_path=ckpt_path)

# Select the test split of the Kitti dataset
test_split = dataset.get_split("test")

# Prepare the visualizer 
vis = Visualizer()

# Variable to accumulate the predictions
data_list = []

# Let's detect objects in the first few point clouds of the Kitti set
for idx in tqdm(range(10)):
	# Get one test point cloud from the SemanticKitti dataset
    data = test_split.get_data(idx)
    
    # Run the inference
    result = pipeline.run_inference(data)[0]
    # Filter out results with low confidence
    result = filter_detections(result)

    # Prepare a dictionary usable by the visulization tool
    pred = {
    "name": 'KITTI' + '_' + str(idx),
    'points': data['point'],
    'bounding_boxes': result
    }

	# Append the data to the list    
    data_list.append(pred)
   
    
# Let's detect objects in the first few point clouds of the custom set
for idx in tqdm(range(len(custom_dataset))):
	# Get one point cloud and format it for inference
    data, pcd = prepare_point_cloud_for_inference(custom_dataset[idx])
 
    # Run the inference
    result = pipeline.run_inference(data)[0]
    # Filter out results with low confidence
    result = filter_detections(result, min_conf = 0.3)
    
    # Prepare a dictionary usable by the visulization tool
    pred = {
    "name": 'Custom' + '_' + str(idx),
    'points': data['point'],
    'bounding_boxes': result
    }

    # Append the data to the list  
    data_list.append(pred)
    

# Visualize the results
vis.visualize(data_list, None, bounding_boxes=None)