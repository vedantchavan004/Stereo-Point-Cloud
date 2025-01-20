import cv2
import numpy as np
from matplotlib import pyplot as plt
import os

# Function to create point cloud file
def create_output(vertices, colors, filename):
    colors = colors.reshape(-1, 3)
    vertices = np.hstack([vertices.reshape(-1, 3), colors])
    ply_header = '''ply
    format ascii 1.0
    element vertex %(vert_num)d
    property float x
    property float y
    property float z
    property uchar red
    property uchar green
    property uchar blue
    end_header
    '''
    with open(filename, 'w') as f:
        f.write(ply_header % dict(vert_num=len(vertices)))
        np.savetxt(f, vertices, '%f %f %f %d %d %d')

# Load calibration data from calib.txt
def load_calibration(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    P0 = np.array([float(x) for x in lines[0].split()[1:]]).reshape(3, 4)
    P1 = np.array([float(x) for x in lines[1].split()[1:]]).reshape(3, 4)
    return P0, P1

# Compute the Q matrix
def compute_q_matrix(P0, P1):
    baseline = -(P1[0, 3] / P1[0, 0])  # Baseline is Tx (translation along x) from P1
    cx = P0[0, 2]
    cy = P0[1, 2]
    focal_length = P0[0, 0]
    
    Q = np.array([
        [1, 0, 0, -cx],
        [0, 1, 0, -cy],
        [0, 0, 0, focal_length],
        [0, 0, -1 / baseline, 0]
    ])
    return Q

# Load stereo images
imgL_path = 'L.png'
imgR_path = 'R.png'

imgL = cv2.imread(imgL_path, cv2.IMREAD_GRAYSCALE)
imgR = cv2.imread(imgR_path, cv2.IMREAD_GRAYSCALE)

# Check if images are loaded
if imgL is None:
    raise FileNotFoundError(f"Left image not found at {imgL_path}. Check the file path.")
if imgR is None:
    raise FileNotFoundError(f"Right image not found at {imgR_path}. Check the file path.")

# Load calibration
calib_file = 'calib.txt'
if not os.path.exists(calib_file):
    raise FileNotFoundError(f"Calibration file not found: {calib_file}")
P0, P1 = load_calibration(calib_file)
Q = compute_q_matrix(P0, P1)

# Create the StereoSGBM object with fine-tuned parameters
stereo = cv2.StereoSGBM_create(
    minDisparity=0,
    numDisparities=16 * 10,  # Increased disparity range (10 x 16)
    blockSize=3,            # Reduced block size for finer details
    uniquenessRatio=5,      # Lower uniqueness ratio for denser matches
    speckleWindowSize=50,   # Smaller window to retain more points
    speckleRange=32,        # Allowable range of speckle disparity
    disp12MaxDiff=1,        # Small max difference between left-right disparities
    P1=8 * 3 * 3 ** 2,      # Penalty for small disparity changes
    P2=32 * 3 * 3 ** 2      # Penalty for large disparity changes
)

# Compute disparity map
disparity = stereo.compute(imgL, imgR).astype(np.float32) / 16.0

# Optionally smooth disparity with a median filter
disparity = cv2.medianBlur(disparity, 5)

# Show disparity map
plt.imshow(disparity, 'gray')
plt.colorbar()
plt.title("Disparity Map")
plt.show()

# Normalize disparity for coloring
disparity_normalized = cv2.normalize(disparity, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
disparity_normalized = disparity_normalized.astype(np.uint8)

# Generate 3D point cloud
print("\nGenerating the 3D map...")
points_3D = cv2.reprojectImageTo3D(disparity, Q)

# Invert the x-axis to fix the left-right flipping issue
points_3D[:, :, 0] = -points_3D[:, :, 0]

# Create mask to filter out invalid disparity values
mask = disparity > disparity.min()

# Get valid points
output_points = points_3D[mask]

# Use the normalized disparity map for color information
output_colors = cv2.applyColorMap(disparity_normalized, cv2.COLORMAP_JET)[mask]

# Save point cloud
output_file = 'point_cloud_with_disparity_fixed.ply'
create_output(output_points, output_colors, output_file)
print(f"Point cloud saved to {output_file}")
