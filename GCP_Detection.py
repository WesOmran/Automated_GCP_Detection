import numpy as np
import pandas as pd
import laspy
from sklearn.cluster import DBSCAN
import plotly.graph_objects as go
import os
import glob

"""
    This Program helps automating the detection of the mid-point in (x, y, z) 
    of GCP markers. This is specifically made for GCP markers that are as follows: 
    +-------------+ 
    |\-----------/|     The GCP that is to be detected is of this shape and color.
    |-\--green--/-|
    |--\-------/--|     The program requires a csv file containing the conrol GCP
    |---\-----/---|     marker positions in (x, y, z) or (Easting, Northing, Elevation)
    |----\---/----|
    |-red-\-/-----|     It creates a buffer zone of whatever is specified by the user in m
    |-----/-\-red-|     around the control GCP marker position. This allows the program to
    |----/---\----|     search for GCP markers in the LAS file a lot more efficiently.
    |---/-----\---|     In that buffer zone, the then starts looping through each point
    |--/-------\--|     to check their RGB values. Using DBScan algorithm, it helps create 
    |-/--green--\-|     clusters that identifies the GCP markers. 
    +/-----------\|     From there the program computes the Euclidean mid-point in (x, y, z).
    
"""

# ------------------- DATA LOADING -------------------
def load_gcp_csv(gcp_csv_path):
    return pd.read_csv(gcp_csv_path)

def load_las_file(las_file_path):
    las = laspy.read(las_file_path)
    points = np.vstack((las.x, las.y, las.z)).T
    colors = np.vstack((las.red, las.green, las.blue)).T
    colors_8bit = colors // 256
    if hasattr(las, 'classification'):
        classification = las.classification
        manual_GCP_mask = classification == 8  # Model Key-point class
        manual_GCP = points[manual_GCP_mask]
    else:
        manual_GCP = np.empty((0, 3))
        print("LAS file does not contain Key-Point Classified data.")
    return points, colors_8bit, manual_GCP

# ------------------- GCP DETECTION -------------------
def detect_gcps(points, colors_8bit, control_GCP, manual_GCP, buffer, z_threshold, eps, min_samples):
    detected_coords = []
    errors = []
    all_red_points = []
    all_green_points = []
    for idx, row in control_GCP.iterrows():
        print(f"\nProcessing {idx + 1}/{control_GCP.shape[0]} GCPs...")
        x0, y0 = row['Easting'], row['Northing']
        mask = (
            (points[:, 0] >= x0 - buffer) & (points[:, 0] <= x0 + buffer) &
            (points[:, 1] >= y0 - buffer) & (points[:, 1] <= y0 + buffer)
        )
        cropped_points = points[mask]
        cropped_colors = colors_8bit[mask]

        if cropped_points.shape[0] == 0:
            print(f"{row['Label']}: No points.")
            detected_coords.append([np.nan, np.nan, np.nan])
            errors.append(np.nan)
            continue

        ground_z = np.percentile(cropped_points[:, 2], 10)
        ground_mask = cropped_points[:, 2] < (ground_z + z_threshold)
        ground_points = cropped_points[ground_mask]
        ground_colors = cropped_colors[ground_mask]

        red_mask = (ground_colors[:, 0] > 120) & (ground_colors[:, 1] < 100)
        green_mask = (ground_colors[:, 1] > 120) & (ground_colors[:, 0] < 100)

        red_points = ground_points[red_mask]
        green_points = ground_points[green_mask]
        # Collect all red/green points
        all_red_points.append(red_points)
        all_green_points.append(green_points)

        rg_points = np.vstack((red_points, green_points))

        if rg_points.shape[0] == 0:
            print(f"{row['Label']}: No red/green.")
            detected_coords.append([np.nan, np.nan, np.nan])
            errors.append(np.nan)
            continue

        labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(rg_points[:, :2])
        valid_labels = np.unique(labels[labels != -1])
        if len(valid_labels) == 0:
            print(f"{row['Label']}: No clusters.")
            detected_coords.append([np.nan, np.nan, np.nan])
            errors.append(np.nan)
            continue

        largest_label = max(valid_labels, key=lambda lbl: np.sum(labels == lbl))
        cluster_points = rg_points[labels == largest_label]
        centroid = np.mean(cluster_points, axis=0)
        detected_coords.append(centroid)

        # Error calculation
        if manual_GCP.shape[0] > 0:
            dists = np.linalg.norm(manual_GCP - centroid[:3], axis=1)
            nearest_idx = np.argmin(dists)
            ref = manual_GCP[nearest_idx]
            dx = float(centroid[0]) - float(ref[0])
            dy = float(centroid[1]) - float(ref[1])
            dz = float(centroid[2]) - float(ref[2])
            error = np.sqrt(dx**2 + dy**2 + dz**2)
            errors.append(error)
            print(f"{row['Label']}: Detected centroid = {centroid}")
            print(f"Difference from Manual GCP = {error:.3f} m")
        else:
            errors.append(np.nan)
            print(f"{row['Label']}: Detected centroid = {centroid}")

    all_red_points = np.vstack([r for r in all_red_points if r.shape[0] > 0]) if all_red_points else np.empty((0,3))
    all_green_points = np.vstack([g for g in all_green_points if g.shape[0] > 0]) if all_green_points else np.empty((0,3))
    return detected_coords, errors, all_red_points, all_green_points

# ------------------- SAVE RESULTS -------------------
def save_detected_gcps(detected_coords, las_file_path):
    detected_gcp_df = pd.DataFrame({
        'Detected_Easting': [c[0] for c in detected_coords],
        'Detected_Northing': [c[1] for c in detected_coords],
        'Detected_Elevation': [c[2] for c in detected_coords],
    })
    detected_export_path = las_file_path.replace(".las", "_DetectedGCPs.csv")
    detected_gcp_df.to_csv(detected_export_path, index=False)
    print(f"Detected GCPs exported to: {detected_export_path}")

# ------------------- VISUALIZATION -------------------
def visualize_detection_interactive(points, red_points, green_points, control_GCP, manual_GCP, detected_coords, max_points):
    margin = 10  # meters
    x_min, x_max = control_GCP['Easting'].min() - margin, control_GCP['Easting'].max() + margin
    y_min, y_max = control_GCP['Northing'].min() - margin, control_GCP['Northing'].max() + margin
    mask = (
        (points[:, 0] >= x_min) & (points[:, 0] <= x_max) &
        (points[:, 1] >= y_min) & (points[:, 1] <= y_max)
    )
    cropped_points = points[mask]
    step = max(1, len(cropped_points) // max_points)
    subsampled = cropped_points[::step]
    fig = go.Figure()
    fig.add_trace(go.Scattergl(
        x=subsampled[:, 0], y=subsampled[:, 1], mode='markers',
        marker=dict(size=1, color='lightgray'), hoverinfo='skip', name='LiDAR (subsampled)'
    ))
    if red_points.shape[0] > 0:
        fig.add_trace(go.Scattergl(
            x=red_points[:, 0], y=red_points[:, 1], mode='markers',
            marker=dict(size=3, color='red'), name='Filtered Red Points'
        ))
    if green_points.shape[0] > 0:
        fig.add_trace(go.Scattergl(
            x=green_points[:, 0], y=green_points[:, 1], mode='markers',
            marker=dict(size=3, color='green'), name='Filtered Green Points'
        ))
    fig.add_trace(go.Scattergl(
        x=control_GCP['Easting'], y=control_GCP['Northing'], mode='markers',
        marker=dict(size=10, color='blue', symbol='x'), name='GCPs (Ground Truth)'
    ))
    detected_coords = np.array(detected_coords)
    fig.add_trace(go.Scattergl(
        x=detected_coords[:, 0], y=detected_coords[:, 1], mode='markers',
        marker=dict(size=10, color='cyan', line=dict(width=1, color='black')), name='Detected GCPs'
    ))
    if manual_GCP.shape[0] > 0:
        fig.add_trace(go.Scattergl(
            x=manual_GCP[:, 0], y=manual_GCP[:, 1], mode='markers',
            marker=dict(size=10, color='magenta', line=dict(width=1, color='black')), name='Model Key-points'
        ))
    fig.update_layout(
        title='GCP Detection Visualization (Interactive)',
        xaxis_title='Easting', yaxis_title='Northing',
        legend=dict(font=dict(size=10)), width=1000, height=800, template='plotly_white'
    )
    print("Launching interactive visualization...")
    fig.show(renderer="browser") 
    
# ------------------- STATISTICAL TESTING -------------------


# ------------------- USER INPUT -------------------
def get_las_files():
    print("Choose LAS file input method:")
    print("1. Enter a folder path containing all LAS files")
    print("2. Enter a comma-separated list of LAS file paths")
    method = input("Enter 1 or 2: ").strip()
    las_files = []
    if (method == "1"):
        folder = input("Enter folder path containing LAS files: ").strip()
        las_files = glob.glob(os.path.join(folder, "*.las"))
        print(f"Found {len(las_files)} LAS files in folder.")
    elif (method == "2"):
        files = input("Enter full paths of LAS files, separated by commas: ").strip()
        las_files = [f.strip() for f in files.split(",") if f.strip()]
        print(f"Received {len(las_files)} LAS files.")
    else:
        print("Invalid input. Exiting.")
        exit(1)
    if not las_files:
        print("No LAS files found. Exiting.")
        exit(1)
    return las_files

# ------------------- MAIN -------------------
def main(buffer, z_threshold, eps, min_samples, max_points):
    print("Program started...")
    las_files = get_las_files()
    for las_file_path in las_files:
        print(f"\nProcessing LAS file: {las_file_path}")
        # Automatically find the corresponding CSV
        gcp_csv_path = os.path.splitext(las_file_path)[0] + ".csv"
        if not os.path.exists(gcp_csv_path):
            print(f"WARNING: Control GCP CSV not found for {las_file_path} (expected: {gcp_csv_path}). Skipping.")
            continue
        control_GCP = load_gcp_csv(gcp_csv_path)
        print(f"GCP CSV loaded: {gcp_csv_path}, shape: {control_GCP.shape}")

        points, colors_8bit, manual_GCP = load_las_file(las_file_path)
        print(f"LAS file loaded: {las_file_path}, points: {points.shape}, manual GCPs: {manual_GCP.shape}")

        if manual_GCP.shape[0] != 0 and manual_GCP.shape[0] != control_GCP.shape[0]:
            print("Warning: Manual GCP count doesn't match control GCP count!")
            manual_GCP = manual_GCP[:control_GCP.shape[0]]
        print("Detecting GCPs...")
        detected_coords, errors, red_points, green_points = detect_gcps(
            points, colors_8bit, control_GCP, manual_GCP, buffer, z_threshold, eps, min_samples
        )
        print("\nSaving detected GCPs...")
        save_detected_gcps(detected_coords, las_file_path)
        print("Visualizing results...")
        visualize_detection_interactive(
            points, red_points, green_points, control_GCP, manual_GCP, detected_coords, max_points
        )
    print("All LAS files processed successfully.")

if __name__ == "__main__":
    # ------------------- PARAMETERS -------------------
    BUFFER = 3.0        # meters around each GCP
    Z_THRESHOLD = 0.5   # height threshold to remove vegetation
    EPS = 0.1           # DBSCAN radius
    MIN_SAMPLES = 10    # DBSCAN minimum cluster size
    MAX_POINTS = 1000   # for visualization subsampling
    
    main(BUFFER, Z_THRESHOLD, EPS, MIN_SAMPLES, MAX_POINTS)