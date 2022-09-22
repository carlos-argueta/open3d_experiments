import os
import open3d as o3d
import open3d.ml as _ml3d
import open3d.ml.torch as ml3d

import numpy as np

import glob

def custom_draw_geometry(pcd):
	vis = o3d.visualization.Visualizer()
	vis.create_window()
	vis.get_render_option().point_size = 2.0
	vis.get_render_option().background_color = np.asarray([1.0, 1.0, 1.0])
	vis.add_geometry(pcd)
	vis.run()
	vis.destroy_window()

def load_custom_dataset(dataset_path):
	print("Loading custom dataset")
	pcd_paths = glob.glob(dataset_path+"/*.pcd")
	pcds = []
	for pcd_path in pcd_paths:
		pcds.append(o3d.io.read_point_cloud(pcd_path))
	return pcds


def prepare_point_cloud_for_inference(pcd):
	# Remove NaNs and infinity values
	pcd.remove_non_finite_points()
	# Extract the xyz points
	xyz = np.asarray(pcd.points)
	# Set the points to the correct format for inference
	data = {"point":xyz, 'feat': None, 'label':np.zeros((len(xyz),), dtype=np.int32)}

	return data, pcd

def load_point_cloud_for_inference(file_path, dataset_path):
	pcd_path = dataset_path + "/" + file_path
	# Load the file
	pcd = o3d.io.read_point_cloud(pcd_path)
	# Remove NaNs and infinity values
	pcd.remove_non_finite_points()
	# Extract the xyz points
	xyz = np.asarray(pcd.points)
	# Set the points to the correct format for inference
	data = {"point":xyz, 'feat': None, 'label':np.zeros((len(xyz),), dtype=np.int32)}

	return data, pcd

# Class colors, RGB values as ints for easy reading
COLOR_MAP = {
    0: (0, 0, 0),
    1: (245, 150, 100),
    2: (245, 230, 100),
    3: (150, 60, 30),
    4: (180, 30, 80),
    5: (255, 0., 0),
    6: (30, 30, 255),
    7: (200, 40, 255),
    8: (90, 30, 150),
    9: (255, 0, 255),
    10: (255, 150, 255),
    11: (75, 0, 75),
    12: (75, 0., 175),
    13: (0, 200, 255),
    14: (50, 120, 255),
    15: (0, 175, 0),
    16: (0, 60, 135),
    17: (80, 240, 150),
    18: (150, 240, 255),
    19: (0, 0, 255),
}

# Convert class colors to doubles from 0 to 1, as expected by the visualizer
for label in COLOR_MAP:
	COLOR_MAP[label] = tuple(val/255 for val in COLOR_MAP[label])

# Load an ML configuration file
cfg_file = "/home/carlos/Open3D/build/Open3D-ML/ml3d/configs/randlanet_semantickitti.yml"
cfg = _ml3d.utils.Config.load_from_file(cfg_file)

# Load the RandLANet model
model = ml3d.models.RandLANet(**cfg.model)
# Add path to the SemanticKitti dataset and your own custom dataset
cfg.dataset['dataset_path'] = '/media/carlos/SeagateExpansionDrive/kitti/SemanticKitti/'
cfg.dataset['custom_dataset_path'] = './pcds'

# Load the datasets
dataset = ml3d.datasets.SemanticKITTI(cfg.dataset.pop('dataset_path', None), **cfg.dataset)
custom_dataset = load_custom_dataset(cfg.dataset.pop('custom_dataset_path', None))

# Create the ML pipeline
pipeline = ml3d.pipelines.SemanticSegmentation(model, dataset=dataset, device="gpu", **cfg.pipeline)

# Download the weights.
ckpt_folder = "./logs/"
os.makedirs(ckpt_folder, exist_ok=True)
ckpt_path = ckpt_folder + "randlanet_semantickitti_202201071330utc.pth"
randlanet_url = "https://storage.googleapis.com/open3d-releases/model-zoo/randlanet_semantickitti_202201071330utc.pth"
if not os.path.exists(ckpt_path):
    cmd = "wget {} -O {}".format(randlanet_url, ckpt_path)
    os.system(cmd)

# Load the parameters of the model.
pipeline.load_ckpt(ckpt_path=ckpt_path)

# Get one test point cloud from the SemanticKitti dataset
pc_idx = 58 # change the index to get a different point cloud
test_split = dataset.get_split("test")
data = test_split.get_data(pc_idx)

# run inference on a single example.
# returns dict with 'predict_labels' and 'predict_scores'.
result = pipeline.run_inference(data)

# Create a pcd to be visualized 
pcd = o3d.geometry.PointCloud()
xyz = data["point"] # Get the points
pcd.points = o3d.utility.Vector3dVector(xyz)

colors = [COLOR_MAP[clr] for clr in list(result['predict_labels'])] # Get the color associated to each predicted label
pcd.colors = o3d.utility.Vector3dVector(colors) # Add color data to the point cloud

# Create visualization
custom_draw_geometry(pcd)

# Get one test point cloud from the custom dataset
pc_idx = 800 # change the index to get a different point cloud
data, pcd = prepare_point_cloud_for_inference(custom_dataset[pc_idx])


# Run inference
result = pipeline.run_inference(data)

# Colorize the point cloud with predicted labels
colors = [COLOR_MAP[clr] for clr in list(result['predict_labels'])]
pcd.colors = o3d.utility.Vector3dVector(colors)

# Create visualization
custom_draw_geometry(pcd)

# evaluate performance on the test set; this will write logs to './logs'.
#pipeline.run_test()