import numpy as np
import pandas as pd
import laspy
from sklearn.cluster import DBSCAN
import plotly.graph_objects as go
import os
import glob
import scipy.stats
import logging
import pyproj

# ------------------- GLOBAL VARIABLE -------------------
logger = logging.getLogger(__name__)

# ------------------- LOGGING -------------------
def logging_setup(log_path):
    """
    Sets up logging configuration for the script.
    Logs will be saved to 'gcp_detection.log' and also printed to the terminal.
    """
    if not os.path.exists("gcp_detection.log"):
        open("gcp_detection.log", 'w').close()  # Create log file if it doesn't exist
    # Reconfigure logging to use the new log file path
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_path, mode='w'),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler("gcp_detection.log", mode='w'),  # log file
            logging.StreamHandler()  # also print to terminal
        ]
    )
    logger = logging.getLogger(__name__)
    return logger

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
        logger.info("LAS file does not contain Key-Point Classified data.")
    return points, colors_8bit, manual_GCP

# ------------------- CRS CHECKING -------------------
def check_las_crs(las_file_path):
    try:
        las = laspy.read(las_file_path)
        crs = las.header.parse_crs()
        logger.info(f"LAS CRS for {os.path.basename(las_file_path)}: {crs}")
        return crs
    except Exception as e:
        logger.warning(f"Could not read CRS from LAS file: {e}")
        return None

def check_csv_crs(gcp_csv_path):
    # This is a placeholder: CSVs don't store CRS natively.
    # You must know or document the CRS used for the CSV.
    # Here, we just log a reminder.
    logger.info(f"Assuming CSV CRS for {os.path.basename(gcp_csv_path)} is EPSG:5550 (update if different).")

# ------------------- GCP DETECTION -------------------
# --- Add this transformer at the top-level (after imports) ---
# Transformer: WGS84 ellipsoid (EPSG:4979) to WGS84+EGM96 geoid (EPSG:4326+5773)
egm96_transformer = pyproj.Transformer.from_crs(
    "EPSG:4979", "EPSG:4326+5773", always_xy=True
)

def correct_z_with_egm96(easting, northing, ellip_height, utm_zone=54, southern_hemisphere=True):
    """
    Converts UTM easting/northing and ellipsoidal height to orthometric height using EGM96.
    """
    utm_crs = pyproj.CRS.from_proj4(
        f"+proj=utm +zone={utm_zone} +{'south' if southern_hemisphere else 'north'} +datum=WGS84 +units=m +no_defs"
    )
    lon, lat = pyproj.Transformer.from_crs(utm_crs, "EPSG:4326", always_xy=True).transform(easting, northing)
    lon, lat, ortho_height = egm96_transformer.transform(lon, lat, ellip_height)
    return ortho_height

# --- Modify your detect_gcps function as follows: ---
def detect_gcps(points, colors_8bit, control_GCP, manual_GCP, buffer, z_threshold, eps, min_samples):
    detected_coords = []
    error_from_manual = []
    error_from_control = []
    all_red_points = []
    all_green_points = []
    utm_zone = 54  # Papua New Guinea is in UTM zone 54S
    for idx, row in control_GCP.iterrows():
        logger.info(f"\nProcessing {idx + 1}/{control_GCP.shape[0]} GCPs...")
        x0, y0 = row['Easting'], row['Northing']
        mask = (
            (points[:, 0] >= x0 - buffer) & (points[:, 0] <= x0 + buffer) &
            (points[:, 1] >= y0 - buffer) & (points[:, 1] <= y0 + buffer)
        )
        cropped_points = points[mask]
        cropped_colors = colors_8bit[mask]

        if cropped_points.shape[0] == 0:
            logger.info(f"{row['Label']}: No points.")
            detected_coords.append([np.nan, np.nan, np.nan])
            error_from_control.append(np.nan)
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
            logger.info(f"{row['Label']}: No red/green.")
            detected_coords.append([np.nan, np.nan, np.nan])
            error_from_control.append(np.nan)
            continue

        labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(rg_points[:, :2])
        valid_labels = np.unique(labels[labels != -1])
        if len(valid_labels) == 0:
            logger.info(f"{row['Label']}: No clusters.")
            detected_coords.append([np.nan, np.nan, np.nan])
            error_from_control.append(np.nan)
            continue

        largest_label = max(valid_labels, key=lambda lbl: np.sum(labels == lbl))
        cluster_points = rg_points[labels == largest_label]
        centroid = np.mean(cluster_points, axis=0)

        # --- Apply EGM96 geoid correction to detected Z ---
        ortho_z = correct_z_with_egm96(centroid[0], centroid[1], centroid[2], utm_zone=utm_zone, southern_hemisphere=True)
        detected_coords.append([centroid[0], centroid[1], ortho_z])

        ref_control = np.array([row['Easting'], row['Northing'], row['Elevation']])
        logger.info(f"{row['Label']}: Control GCP = {ref_control}, Detected centroid = {[centroid[0], centroid[1], ortho_z]}")

        # 3D Euclidean error (now using orthometric Z)
        if not np.any(np.isnan(centroid)):
            error_control_3d = np.linalg.norm(np.array([centroid[0], centroid[1], ortho_z]) - ref_control)
            error_control_2d = np.linalg.norm(np.array([centroid[0], centroid[1]]) - ref_control[:2])
            error_from_control.append(error_control_3d)
            logger.info(f"{row['Label']}: 3D Error from Control GCP = {error_control_3d:.3f} m")
            logger.info(f"{row['Label']}: 2D Error from Control GCP = {error_control_2d:.3f} m")
            logger.info(f"{row['Label']}: Z (Detected, ortho) = {ortho_z:.3f}, Z (Control) = {ref_control[2]:.3f}, Z diff = {ortho_z - ref_control[2]:.3f} m")
        else:
            error_from_control.append(np.nan)
            logger.info(f"{row['Label']}: No detected centroid, skipping error calculation.")

        # Error to manual GCP (if present)
        if manual_GCP.shape[0] > 0 and not np.any(np.isnan(centroid)):
            dists = np.linalg.norm(manual_GCP - np.array([centroid[0], centroid[1], ortho_z]), axis=1)
            nearest_idx = np.argmin(dists)
            ref = manual_GCP[nearest_idx]
            error_manual = np.linalg.norm(np.array([centroid[0], centroid[1], ortho_z]) - ref)
            error_from_manual.append(error_manual)
            logger.info(f"{row['Label']}: Difference from Manual GCP = {error_manual:.3f} m")
        else:
            error_from_manual.append(np.nan)

    detected_coords = np.array(detected_coords)
    error_from_control = np.array(error_from_control)
    all_red_points = np.vstack([r for r in all_red_points if r.shape[0] > 0]) if all_red_points else np.empty((0,3))
    all_green_points = np.vstack([g for g in all_green_points if g.shape[0] > 0]) if all_green_points else np.empty((0,3))
    return detected_coords, error_from_control, all_red_points, all_green_points

# ------------------- SAVE RESULTS -------------------
def save_detected_gcps(detected_coords, las_file_path):
    detected_gcp_df = pd.DataFrame({
        'Detected_Easting': detected_coords[:, 0],
        'Detected_Northing': detected_coords[:, 1],
        'Detected_Elevation': detected_coords[:, 2],
    })
    detected_export_path = las_file_path.replace(".las", "_DetectedGCPs.csv")
    detected_gcp_df.to_csv(detected_export_path, index=False)
    logger.info(f"Detected GCPs exported to: {detected_export_path}")

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
    logger.info("Launching interactive visualization...")
    fig.show(renderer="browser")
    
# ------------------- STATISTICAL TESTING -------------------
def test_statistics(errors, threshold, alpha):
    """
    Performs statistical analysis on GCP detection errors:
    - Computes sample mean and std deviation
    - Computes confidence interval for the mean using t-distribution
    - Performs one-sided t-test: H0: mean >= threshold vs Ha: mean < threshold
    - Flags outliers (>2 std dev from mean)
    """

    # Step 1: Compute Euclidean errors 
    mean_error = np.mean(errors)
    std_error = np.std(errors, ddof=1)
    n = len(errors)
    df = n - 1

    logger.info(f"\n--- GCP Detection Error Statistics ---")
    logger.info(f"Mean Error: {mean_error:.4f} m")
    logger.info(f"Standard Deviation: {std_error:.4f} m")

    # Step 2: Confidence Interval
    t_crit = scipy.stats.t.ppf(1 - alpha/2, df)
    ci_half_width = t_crit * (std_error / np.sqrt(n))
    ci = [mean_error - ci_half_width, mean_error + ci_half_width]
    logger.info(f"{int((1-alpha)*100)}% Confidence Interval: [{ci[0]:.4f}, {ci[1]:.4f}] m")

    # Step 3: Hypothesis Test (one-sided, mean < threshold)
    t_stat = (mean_error - threshold) / (std_error / np.sqrt(n))
    p_value = scipy.stats.t.cdf(t_stat, df)  # one-sided, lower tail

    logger.info(f"\nHypothesis Test: H0: mean >= {threshold:.2f} m, Ha: mean < {threshold:.2f} m")
    logger.info(f"t statistic: {t_stat:.4f}, p-value: {p_value:.4f}")

    passed = p_value < alpha

    if passed:
        logger.info(f"Result: REJECT H0 (mean error is statistically < {threshold:.2f} m at alpha={alpha})")
    else:
        logger.info(f"Result: FAIL TO REJECT H0 (mean error is NOT statistically < {threshold:.2f} m at alpha={alpha})")

    # Step 4: Outlier Detection
    outlier_indices = np.where(np.abs(errors - mean_error) > 2 * std_error)[0]
    if len(outlier_indices) > 0:
        logger.info(f"\nOutlier GCPs (>2 std dev from mean):")
        for idx in outlier_indices:
            logger.info(f"\tGCP {idx+1}: Error = {errors[idx]:.4f} m")
    else:
        logger.info("\nNo outlier GCPs (>2 std dev from mean).")

    # Optionally, return results for further use
    return {
        "mean": mean_error,
        "std": std_error,
        "ci": ci,
        "t_stat": t_stat,
        "p_value": p_value,
        "outliers": outlier_indices,
        "errors": errors,
        "passed": passed
    }

# ------------------- USER INPUT -------------------
def get_las_files():
    logger.info("Choose LAS file input method:")
    logger.info("1. Enter a folder path containing all LAS files")
    logger.info("2. Enter a comma-separated list of LAS file paths")
    method = input("Enter 1 or 2: ").strip()
    las_files = []
    if (method == "1"):
        folder = input("Enter folder path containing LAS files: ").strip()
        las_files = glob.glob(os.path.join(folder, "*.las"))
        logger.info(f"Found {len(las_files)} LAS files in folder.")
    elif (method == "2"):
        files = input("Enter full paths of LAS files, separated by commas: ").strip()
        las_files = [f.strip() for f in files.split(",") if f.strip()]
        logger.info(f"Received {len(las_files)} LAS files.")
    else:
        logger.info("Invalid input. Exiting.")
        exit(1)
    if not las_files:
        logger.info("No LAS files found. Exiting.")
        exit(1)
    return las_files

# ------------------- MAIN -------------------
def main(buffer, z_threshold, eps, min_samples, max_points, alpha, threshold):
    # Ask for LAS folder or get from first LAS file
    las_files = get_las_files()
    if not las_files:
        print("No LAS files found. Exiting.")
        return
    log_folder = os.path.dirname(las_files[0])
    log_path = os.path.join(log_folder, "gcp_detection.log")

    logger = logging_setup(log_path)

    logger.info("Program started...")
    
    # Check CRS for first LAS and CSV
    check_las_crs(las_files[0])
    # Find corresponding CSV for first LAS file
    gcp_csv_path = os.path.splitext(las_files[0])[0] + ".csv"
    if os.path.exists(gcp_csv_path):
        check_csv_crs(gcp_csv_path)
    else:
        logger.warning(f"No CSV found for CRS check: {gcp_csv_path}")

    # Load LAS files
    las_files = get_las_files()
    
    # Process each LAS file
    for las_file_path in las_files:
        logger.info(f"\nProcessing LAS file: {las_file_path}")
        # Automatically find the corresponding CSV
        gcp_csv_path = os.path.splitext(las_file_path)[0] + ".csv"
        if not os.path.exists(gcp_csv_path):
            logger.info(f"WARNING: Control GCP CSV not found for {las_file_path} (expected: {gcp_csv_path}). Skipping.")
            continue
        control_GCP = load_gcp_csv(gcp_csv_path)
        logger.info(f"GCP CSV loaded: {gcp_csv_path}\n\tshape: {control_GCP.shape}")

        # Load LAS file
        points, colors_8bit, manual_GCP = load_las_file(las_file_path)
        logger.info(f"LAS file loaded: {las_file_path}\n\tpoints: {points.shape}\n\tmanual GCPs: {manual_GCP.shape}")

        # Check if manual GCPs are present
        if manual_GCP.shape[0] != 0 and manual_GCP.shape[0] != control_GCP.shape[0]:
            logger.info("Warning: Manual GCP count doesn't match control GCP count!")
            manual_GCP = manual_GCP[:control_GCP.shape[0]]

        # Detect GCPs
        logger.info("Detecting GCPs...")
        detected_coords, error_from_control, red_points, green_points = detect_gcps(
            points, colors_8bit, control_GCP, manual_GCP, buffer, z_threshold, eps, min_samples
        )
        
        # Run statistical tests
        stats = test_statistics(error_from_control, threshold, alpha)
        if stats["passed"]:
            logger.info("Statistical test PASSED: Saving all detected GCPs.")
            save_detected_gcps(detected_coords, las_file_path)
        else:
            logger.info("Statistical test FAILED: Filtering out GCPs with error >= threshold before saving.")
            keep_mask = error_from_control < threshold
            filtered_detected = detected_coords[keep_mask]
            if filtered_detected.shape[0] == 0:
                logger.info("No GCPs passed the error threshold. Nothing will be saved.")
            else:
                logger.info(f"Filtered detected GCPs: {filtered_detected.shape[0]} out of {detected_coords.shape[0]}")
                save_detected_gcps(filtered_detected, las_file_path)

        # Visualize results
        logger.info("Visualizing results...")
        visualize_detection_interactive(
            points, red_points, green_points, control_GCP, manual_GCP, detected_coords, max_points
        )
    logger.info("All LAS files processed successfully.")

if __name__ == "__main__":
    # ------------------- PARAMETERS -------------------
    # GCP detection parameters
    BUFFER = 3.0        # meters around each GCP
    Z_THRESHOLD = 0.5   # height threshold to remove vegetation
    
    # DBSCAN parameters
    EPS = 0.1           # DBSCAN radius
    MIN_SAMPLES = 100    # DBSCAN minimum cluster size
    
    # Visualization parameters
    MAX_POINTS = 1000   # for visualization subsampling
    
    # Statistical testing parameters
    ALPHA = 0.05        # significance level
    THRESHOLD = 0.1     # threshold for hypothesis testing in meters
    
    main(BUFFER, Z_THRESHOLD, EPS, MIN_SAMPLES, MAX_POINTS, ALPHA, THRESHOLD)