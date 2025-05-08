import streamlit as st
import cv2
import numpy as np
from osgeo import gdal # Make sure GDAL is correctly installed in your environment
from shapely.geometry import Point # Not explicitly used in your provided functions but was in imports
import matplotlib.pyplot as plt # For plotting within Streamlit if needed, though st.image is often enough

# --- Your Existing Functions (with minor adjustments for Streamlit) ---

# Function to process the raster image
def process_raster_image(image_bytes):
    # Save the uploaded bytes to a temporary file because gdal.Open typically needs a path
    # or use a GDAL in-memory approach if available and preferred (e.g., with /vsimem/)
    temp_image_path = "temp_uploaded_image.png" # Or .tif if that's the expected format
    with open(temp_image_path, "wb") as f:
        f.write(image_bytes)

    ds = gdal.Open(temp_image_path)
    if ds is None:
        st.error("Failed to open the image with GDAL. Please ensure it's a valid raster format.")
        return None, None, None

    geotransform = ds.GetGeoTransform()
    if geotransform is None:
        st.warning("Failed to get geotransform information. Geographic coordinates might be incorrect or unavailable.")
        # You might want to decide if you proceed with pixel coordinates only or stop
        # For now, let's assume we might proceed but flag it.
        # geotransform = (0, 1, 0, 0, 0, 1) # Default to pixel coords if no geotransform

    image_band = ds.GetRasterBand(1)
    image_array = image_band.ReadAsArray()

    # Normalize image values to [0, 255] for OpenCV processing
    if np.max(image_array) == np.min(image_array): # Avoid division by zero for blank images
        image_norm = np.zeros_like(image_array, dtype=np.uint8)
    else:
        image_norm = ((image_array - np.min(image_array)) / (np.max(image_array) - np.min(image_array)) * 255).astype(np.uint8)

    edges = cv2.Canny(image_norm, 100, 200)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        st.warning("No contours found in the image.")
        return image_norm, [], [] # Return original image and empty lists

    approx_polygons = []
    for contour in contours:
        epsilon = 0.01 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        approx_polygons.append(approx)

    corner_points_pixel = []
    for polygon in approx_polygons:
        for point in polygon:
            corner_points_pixel.append(tuple(point[0]))

    corner_points_filtered_pixel = []
    if corner_points_pixel: # Only filter if there are points
        # Remove overlapping points (basic approach)
        for point in corner_points_pixel:
            if not corner_points_filtered_pixel or \
               all(np.linalg.norm(np.array(point) - np.array(existing_point)) > 10 for existing_point in corner_points_filtered_pixel):
                corner_points_filtered_pixel.append(point)

        corner_points_filtered_pixel.sort(key=lambda p: (p[1], p[0])) # Sort by y, then x

    # Convert pixel coordinates to geographic coordinates
    corner_points_geocoords_with_numbers = []
    if geotransform: # Only if geotransform is available
        for i, point in enumerate(corner_points_filtered_pixel, start=1):
            x_geo = geotransform[0] + point[0] * geotransform[1] + point[1] * geotransform[2]
            y_geo = geotransform[3] + point[0] * geotransform[4] + point[1] * geotransform[5]
            corner_points_geocoords_with_numbers.append({'id': i, 'pixel': point, 'geo': (x_geo, y_geo)})
    else: # Fallback to pixel coordinates if no geotransform
         for i, point in enumerate(corner_points_filtered_pixel, start=1):
            corner_points_geocoords_with_numbers.append({'id': i, 'pixel': point, 'geo': ('N/A', 'N/A')})


    # Create an image with corner points drawn (for display)
    output_image_display = cv2.cvtColor(image_norm, cv2.COLOR_GRAY2BGR) # Convert to BGR for color drawing
    for i, point_data in enumerate(corner_points_geocoords_with_numbers):
        px, py = point_data['pixel']
        cv2.circle(output_image_display, (int(px), int(py)), 5, (0, 0, 255), -1)  # Red circle
        cv2.putText(output_image_display, str(point_data['id']), (int(px) + 7, int(py) + 7),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)  # Blue text

    return image_norm, output_image_display, corner_points_geocoords_with_numbers


def calculate_euclidean_distance(points_data, point1_id, point2_id):
    point1_data = next((p for p in points_data if p['id'] == point1_id), None)
    point2_data = next((p for p in points_data if p['id'] == point2_id), None)

    if not point1_data or not point2_data:
        return None

    # Calculate distance using geographic coordinates if available and valid
    if point1_data['geo'] != ('N/A', 'N/A') and point2_data['geo'] != ('N/A', 'N/A'):
        p1_coord = point1_data['geo']
        p2_coord = point2_data['geo']
        unit_label = "geo units"
    else: # Fallback to pixel coordinates
        st.warning("Calculating distance using pixel coordinates as geographic coordinates are unavailable for one or both points.")
        p1_coord = point1_data['pixel']
        p2_coord = point2_data['pixel']
        unit_label = "pixels"

    distance = np.sqrt((p1_coord[0] - p2_coord[0])**2 + (p1_coord[1] - p2_coord[1])**2)
    return distance, unit_label

# --- Streamlit App Layout ---
st.set_page_config(layout="wide")
st.title("Image Point Extractor & Distance Calculator")

st.sidebar.header("Upload Image")
uploaded_file = st.sidebar.file_uploader("Choose a raster image file (e.g., PNG, TIF)", type=["png", "tif", "tiff", "jpg", "jpeg"])

# Initialize session state variables
if 'processed_points' not in st.session_state:
    st.session_state.processed_points = []
if 'original_image_display' not in st.session_state:
    st.session_state.original_image_display = None
if 'output_image_display' not in st.session_state:
    st.session_state.output_image_display = None


if uploaded_file is not None:
    image_bytes = uploaded_file.getvalue()

    # Display uploaded image
    st.sidebar.image(image_bytes, caption="Uploaded Image", use_column_width=True)

    if st.sidebar.button("Process Image"):
        with st.spinner("Processing image..."):
            original_img_array, output_img_processed, points_data = process_raster_image(image_bytes)
            if original_img_array is not None and points_data is not None:
                st.session_state.original_image_display = original_img_array
                st.session_state.output_image_display = output_img_processed
                st.session_state.processed_points = points_data
                st.success(f"Image processed! Found {len(points_data)} corner points.")
            else:
                st.error("Image processing failed.")
                st.session_state.processed_points = []
                st.session_state.original_image_display = None
                st.session_state.output_image_display = None


col1, col2 = st.columns(2)

with col1:
    st.subheader("Original Image (Grayscale)")
    if st.session_state.original_image_display is not None:
        st.image(st.session_state.original_image_display, caption="Original Grayscale Image", use_column_width=True)
    else:
        st.info("Upload and process an image to see the original.")

with col2:
    st.subheader("Image with Detected Corner Points")
    if st.session_state.output_image_display is not None:
        st.image(st.session_state.output_image_display, caption="Processed Image with Points", use_column_width=True, channels="BGR")
    else:
        st.info("Upload and process an image to see the results.")


st.divider()

if st.session_state.processed_points:
    st.subheader("Detected Corner Points")
    points_info = []
    for p_data in st.session_state.processed_points:
        geo_coords_str = f"({p_data['geo'][0]:.2f}, {p_data['geo'][1]:.2f})" if p_data['geo'] != ('N/A', 'N/A') else "N/A"
        points_info.append({
            "Point ID": p_data['id'],
            "Pixel Coords (X, Y)": f"({p_data['pixel'][0]}, {p_data['pixel'][1]})",
            "Geo Coords (Lon/X, Lat/Y)": geo_coords_str
        })
    st.dataframe(points_info, use_container_width=True)


    st.subheader("Calculate Distance Between Two Points")
    point_ids = [p['id'] for p in st.session_state.processed_points]

    if len(point_ids) >= 2:
        col_dist1, col_dist2 = st.columns(2)
        with col_dist1:
            point1_id_select = st.selectbox("Select First Point ID:", options=point_ids, key="p1_select")
        with col_dist2:
            point2_id_select = st.selectbox("Select Second Point ID:", options=point_ids, key="p2_select", index=min(1, len(point_ids)-1))

        if st.button("Calculate Distance"):
            if point1_id_select == point2_id_select:
                st.warning("Please select two different points.")
            else:
                distance, unit = calculate_euclidean_distance(st.session_state.processed_points, point1_id_select, point2_id_select)
                if distance is not None:
                    st.success(f"Euclidean distance between Point {point1_id_select} and Point {point2_id_select}: **{distance:.2f} {unit}**")
                else:
                    st.error("Could not calculate distance. Point ID not found.")
    elif point_ids:
         st.warning("Need at least two points to calculate distance.")
    else:
        st.info("No points detected to calculate distance.")

else:
    st.info("Upload and process an image to detect points and calculate distances.")

st.sidebar.markdown("---")
st.sidebar.info("This app uses GDAL, OpenCV, and Streamlit.")
