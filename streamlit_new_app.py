import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import supervision as sv
from PIL import Image, ImageOps
import os
import tempfile

# Set page configuration
st.set_page_config(
    page_title="Two-Model Detection Pipeline",
    page_icon="🎯",
    layout="wide"
)

# Title and description
st.title("🎯 Two-Model Detection Pipeline")
st.markdown("**best1.pt** → **Crop Region** → **vstrip.pt** → **Vertical Strip Detection**")

# Sidebar for model configuration
st.sidebar.header("⚙️ Model Configuration")

# Model paths
best1_model_path = st.sidebar.text_input(
    "Best1 Model Path", 
    value="runs/obb/train/weights/best1.pt",
    help="Path to best1.pt model weights"
)

vstrip_model_path = st.sidebar.text_input(
    "VStrip Model Path", 
    value="runs/obb/train/weights/vstrip.pt",
    help="Path to vstrip.pt model weights"
)

# Detection parameters
conf_threshold = st.sidebar.slider(
    "Confidence Threshold", 
    min_value=0.0, 
    max_value=1.0, 
    value=0.25, 
    step=0.05,
    help="Minimum confidence score for detections"
)

iou_threshold = st.sidebar.slider(
    "IoU Threshold", 
    min_value=0.0, 
    max_value=1.0, 
    value=0.45, 
    step=0.05,
    help="IoU threshold for Non-Maximum Suppression"
)

img_size = st.sidebar.selectbox(
    "Image Size", 
    options=[320, 480, 640, 800, 1024],
    index=2,
    help="Input image size for inference"
)

# Device selection
device = st.sidebar.selectbox(
    "Device",
    options=["cpu", "cuda"],
    index=0,
    help="Device for inference (CPU or CUDA)"
)

# Preprocessing toggle
use_preprocessing = st.sidebar.checkbox(
    "Apply Image Preprocessing",
    value=True,
    help="Apply background normalization preprocessing to improve detection accuracy"
)

@st.cache_resource
def load_model(model_path):
    """Load YOLOv8 model with caching"""
    try:
        model = YOLO(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

def preprocess_image(image_path):
    """Apply image preprocessing to improve detection"""
    try:
        # Read image
        img = cv2.imread(image_path, -1)
        
        # Split into RGB planes
        rgb_planes = cv2.split(img)
        
        result_planes = []
        result_norm_planes = []
        for plane in rgb_planes:
            dilated_img = cv2.dilate(plane, np.ones((7,7), np.uint8))
            bg_img = cv2.medianBlur(dilated_img, 21)
            diff_img = 255 - cv2.absdiff(plane, bg_img)
            norm_img = cv2.normalize(diff_img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
            result_planes.append(diff_img)
            result_norm_planes.append(norm_img)
        
        result = cv2.merge(result_planes)
        result_norm = cv2.merge(result_norm_planes)
        
        return result_norm
    except Exception as e:
        st.error(f"Error in preprocessing: {str(e)}")
        return None

def process_image_with_best1(image, model, conf_thresh, iou_thresh, img_sz, device_type, use_preprocessing=True):
    """Process image with best1.pt model and return detections"""
    try:
        # Fix orientation, drop alpha
        image = ImageOps.exif_transpose(image).convert("RGB")

        # Save to tmp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tf:
            tmp_path = tf.name
            image.save(tmp_path, format="JPEG", quality=95)

        # Apply preprocessing if enabled
        preprocessed_path = None
        if use_preprocessing:
            preprocessed_img = preprocess_image(tmp_path)
            if preprocessed_img is not None:
                # Save preprocessed image to new temp file
                with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tf_preprocessed:
                    preprocessed_path = tf_preprocessed.name
                    cv2.imwrite(preprocessed_path, preprocessed_img)
                inference_path = preprocessed_path
            else:
                inference_path = tmp_path
        else:
            inference_path = tmp_path

        # Run inference with best1.pt model - exactly like the working code
        results = model.predict(
            source=inference_path,
            device=device_type,
            half=False,        # keep FP32 on CPU
            imgsz=img_sz,
            conf=conf_thresh,
            iou=iou_thresh,
            verbose=False
        )

        if len(results) > 0:
            return results[0], tmp_path, inference_path
        else:
            return None, tmp_path, inference_path

    except Exception as e:
        st.error(f"Error processing image with best1.pt: {str(e)}")
        return None, None, None

def detect_vstrips_in_crop(crop_image, vstrip_model, device_type, conf_thresh=0.25, iou_thresh=0.45, img_sz=640):
    """Detect vertical strips in a cropped region using vstrip.pt model"""
    try:
        # Save cropped image to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as temp_crop:
            crop_pil = Image.fromarray(crop_image)
            crop_pil.save(temp_crop.name, format='PNG')
            
            # Run vstrip detection - exactly like the working code
            vstrip_results = vstrip_model.predict(
                source=temp_crop.name,
                device=device_type,
                half=False,        # keep FP32 on CPU
                imgsz=img_sz,
                conf=conf_thresh,
                iou=iou_thresh,
                verbose=False
            )
            
            # Debug output for vstrip detection
            st.write(f"Debug: VStrip results type: {type(vstrip_results)}")
            st.write(f"Debug: VStrip results length: {len(vstrip_results) if vstrip_results else 'None'}")
            if vstrip_results and len(vstrip_results) > 0:
                vstrip_result = vstrip_results[0]
                st.write(f"Debug: VStrip first result type: {type(vstrip_result)}")
                if hasattr(vstrip_result, 'obb') and vstrip_result.obb is not None:
                    st.write(f"Debug: VStrip OBB detections: {vstrip_result.obb.xyxyxyxy.shape if hasattr(vstrip_result.obb, 'xyxyxyxy') else 'No xyxyxyxy'}")
                if hasattr(vstrip_result, 'boxes') and vstrip_result.boxes is not None:
                    st.write(f"Debug: VStrip boxes detections: {len(vstrip_result.boxes)}")
                else:
                    st.write("Debug: VStrip boxes is None (normal for OBB models)")
            
            # Clean up temp file
            os.unlink(temp_crop.name)
            
            if vstrip_results is not None and len(vstrip_results) > 0:
                # Get vstrip detections
                vstrip_detections = sv.Detections.from_ultralytics(vstrip_results[0])
                
                # Create visualization with vstrip detections
                vstrip_vis = crop_image.copy()
                vstrip_annotator = sv.OrientedBoxAnnotator(thickness=2, color=sv.ColorPalette.DEFAULT)
                vstrip_annotated = vstrip_annotator.annotate(scene=vstrip_vis, detections=vstrip_detections)
                
                return vstrip_detections, vstrip_annotated, vstrip_results[0]
            else:
                return None, None, None
                
    except Exception as e:
        st.error(f"Error in vstrip detection: {str(e)}")
        return None, None, None

def main():
    # Load models
    if os.path.exists(best1_model_path):
        best1_model = load_model(best1_model_path)
        if best1_model is None:
            st.error("Failed to load best1.pt model. Please check the model path.")
            return
        st.sidebar.success("✅ Best1.pt model loaded successfully!")
    else:
        st.error(f"Best1.pt model file not found at: {best1_model_path}")
        return
    
    if os.path.exists(vstrip_model_path):
        vstrip_model = load_model(vstrip_model_path)
        if vstrip_model is None:
            st.error("Failed to load vstrip.pt model. Please check the model path.")
            return
        st.sidebar.success("✅ VStrip.pt model loaded successfully!")
    else:
        st.error(f"VStrip.pt model file not found at: {vstrip_model_path}")
        return
    
    # File upload
    uploaded_file = st.file_uploader(
        "Choose an image file", 
        type=['png', 'jpg', 'jpeg', 'webp'],
        help="Upload an image to detect objects and vertical strips"
    )
    
    if uploaded_file is not None:
        # Display original image
        image = Image.open(uploaded_file)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📸 Original Image")
            st.image(image, caption="Uploaded Image", use_container_width=True)
            
            # Show preprocessed image if preprocessing is enabled
            if use_preprocessing:
                st.subheader("🔧 Preprocessed Image")
                with st.spinner("Applying preprocessing..."):
                    # Apply preprocessing to show the result
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tf:
                        tmp_path_show = tf.name
                        image.save(tmp_path_show, format="JPEG", quality=95)
                        preprocessed_img = preprocess_image(tmp_path_show)
                        if preprocessed_img is not None:
                            preprocessed_rgb = cv2.cvtColor(preprocessed_img, cv2.COLOR_BGR2RGB)
                            st.image(preprocessed_rgb, caption="Preprocessed Image", use_container_width=True)
                        else:
                            st.warning("Preprocessing failed, using original image")
        
        with col2:
            st.subheader("🎯 Detection Results")
            
            # Process image with best1.pt
            with st.spinner("Running best1.pt inference..."):
                best1_results, tmp_path, inference_path = process_image_with_best1(
                    image, best1_model, conf_threshold, iou_threshold, img_size, device, use_preprocessing
                )
            
            if best1_results is not None:
                # Get detections
                detections = sv.Detections.from_ultralytics(best1_results)
                
                if len(detections) > 0:
                    # Display best1.pt results
                    st.write(f"**Found {len(detections)} object(s) with best1.pt:**")
                    
                    # Create annotated image
                    img_bgr = cv2.imread(tmp_path)
                    annotator = sv.OrientedBoxAnnotator(thickness=2)
                    annotated_frame = annotator.annotate(scene=img_bgr.copy(), detections=detections)
                    annotated_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                    
                    st.image(annotated_rgb, caption="Best1.pt Detections", use_container_width=True)
                    
                    # Process each detection with vstrip.pt
                    st.subheader("📏 Vertical Strip Detection Results")
                    
                    # Get original image for cropping
                    original_img = cv2.imread(tmp_path)
                    if original_img is not None:
                        original_img_rgb = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
                        
                        # Create columns for results
                        num_detections = len(detections)
                        cols_per_row = 2
                        num_rows = (num_detections + cols_per_row - 1) // cols_per_row
                        
                        for row in range(num_rows):
                            cols = st.columns(cols_per_row)
                            for col_idx in range(cols_per_row):
                                detection_idx = row * cols_per_row + col_idx
                                if detection_idx < num_detections:
                                    with cols[col_idx]:
                                        # Get bounding box coordinates
                                        x1, y1, x2, y2 = detections.xyxy[detection_idx].astype(int)
                                        
                                        # Add padding around the crop
                                        padding = 10
                                        x1 = max(0, x1 - padding)
                                        y1 = max(0, y1 - padding)
                                        x2 = min(original_img_rgb.shape[1], x2 + padding)
                                        y2 = min(original_img_rgb.shape[0], y2 + padding)
                                        
                                        # Crop from original image
                                        cropped_img = original_img_rgb[y1:y2, x1:x2]
                                        
                                        # Get detection info
                                        confidence = detections.confidence[detection_idx]
                                        class_id = int(detections.class_id[detection_idx])
                                        
                                        # Get class name if available
                                        if hasattr(best1_results, 'names') and class_id in best1_results.names:
                                            class_name = best1_results.names[class_id]
                                        else:
                                            class_name = f"Class {class_id}"
                                        
                                        st.write(f"**Detection {detection_idx + 1}: {class_name} ({confidence:.3f})**")
                                        
                                        # Run vstrip detection on this crop
                                        with st.spinner(f"Detecting vstrips in {class_name}..."):
                                            vstrip_detections, vstrip_annotated, vstrip_results = detect_vstrips_in_crop(
                                                cropped_img, vstrip_model, device, conf_threshold, iou_threshold, img_size
                                            )
                                        
                                        if vstrip_detections is not None:
                                            st.write(f"**Found {len(vstrip_detections)} vertical strip(s):**")
                                            st.image(vstrip_annotated, caption="VStrip Detections", use_container_width=True)
                                            
                                            # Show vstrip details
                                            vstrip_data = []
                                            for i in range(len(vstrip_detections)):
                                                v_conf = vstrip_detections.confidence[i]
                                                v_class_id = int(vstrip_detections.class_id[i])
                                                v_x1, v_y1, v_x2, v_y2 = vstrip_detections.xyxy[i]
                                                
                                                # Get class name if available
                                                if hasattr(vstrip_results, 'names') and v_class_id in vstrip_results.names:
                                                    v_class_name = vstrip_results.names[v_class_id]
                                                else:
                                                    v_class_name = f"VStrip {v_class_id}"
                                                
                                                vstrip_data.append({
                                                    "Strip #": i + 1,
                                                    "Class": v_class_name,
                                                    "Confidence": f"{v_conf:.3f}",
                                                    "X1": f"{v_x1:.1f}",
                                                    "Y1": f"{v_y1:.1f}",
                                                    "X2": f"{v_x2:.1f}",
                                                    "Y2": f"{v_y2:.1f}"
                                                })
                                            
                                            if vstrip_data:
                                                st.dataframe(vstrip_data, use_container_width=True)
                                        else:
                                            st.info("No vertical strips detected in this crop.")
                                        
                                        # Show original crop
                                        st.write("**Original Crop:**")
                                        st.image(cropped_img, caption="Cropped Region", use_container_width=True)
                else:
                    st.info("No objects detected with best1.pt model.")
            else:
                st.error("Failed to process image with best1.pt model.")
    
    # Instructions
    if uploaded_file is None:
        st.info("👆 Please upload an image to start the two-model detection pipeline.")
        
        st.subheader("ℹ️ How to use:")
        st.markdown("""
        1. **Configure Models**: Set the paths to your best1.pt and vstrip.pt model weights in the sidebar
        2. **Adjust Parameters**: Fine-tune confidence and IoU thresholds as needed
        3. **Upload Image**: Click "Browse files" to upload an image for detection
        4. **View Results**: 
           - Best1.pt detects objects in the full image
           - Each detected region is cropped from the original image
           - VStrip.pt detects vertical strips within each crop
        5. **Analyze**: View both object detections and vertical strip detections with detailed information
        """)

if __name__ == "__main__":
    main()
