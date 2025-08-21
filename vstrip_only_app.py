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
    page_title="VStrip Detection Only",
    page_icon="📏",
    layout="wide"
)

# Title and description
st.title("📏 VStrip Detection Only")
st.markdown("Simple vertical strip detection using **vstrip.pt** model")

# Sidebar for model configuration
st.sidebar.header("⚙️ Model Configuration")

# Model path
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

@st.cache_resource
def load_model(model_path):
    """Load YOLOv8 model with caching"""
    try:
        model = YOLO(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

def detect_vstrips(image, model, conf_thresh, iou_thresh, img_sz, device_type):
    """Detect vertical strips in an image using vstrip.pt model"""
    try:
        # Save uploaded image to a temporary file first
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tf:
            tmp_path = tf.name
            image.save(tmp_path, format="JPEG", quality=95)

        # Run vstrip detection - EXACTLY like the working code
        results = model.predict(
            source=tmp_path,           # Use file path directly like working code
            device=device_type,
            half=False,                # keep FP32 on CPU
            imgsz=img_sz,
            conf=conf_thresh,
            iou=iou_thresh,
            verbose=False
        )
        
        # Debug output
        st.write(f"Debug: Results type: {type(results)}")
        st.write(f"Debug: Results length: {len(results) if results else 'None'}")
        if results and len(results) > 0:
            result = results[0]
            st.write(f"Debug: First result type: {type(result)}")
            if hasattr(result, 'obb') and result.obb is not None:
                st.write(f"Debug: OBB detections: {result.obb.xyxyxyxy.shape if hasattr(result.obb, 'xyxyxyxy') else 'No xyxyxyxy'}")
            if hasattr(result, 'boxes') and result.boxes is not None:
                st.write(f"Debug: Boxes detections: {len(result.boxes)}")
            else:
                st.write("Debug: Boxes is None (this is normal for OBB models)")

        if results is not None and len(results) > 0:
            # Get vstrip detections - exactly like working code
            det = sv.Detections.from_ultralytics(results[0])
            
            # Create visualization - exactly like working code
            frame_bgr = cv2.imread(tmp_path)  # supervision expects BGR np.array; cv2.imread is fine
            annotator = sv.OrientedBoxAnnotator()
            annotated = annotator.annotate(scene=frame_bgr, detections=det)
            
            # Clean up temp file
            os.unlink(tmp_path)
            
            return det, annotated, results[0]
        else:
            # Clean up temp file
            os.unlink(tmp_path)
            return None, None, None

    except Exception as e:
        st.error(f"Error in vstrip detection: {str(e)}")
        # Clean up temp file on error
        if 'tmp_path' in locals():
            os.unlink(tmp_path)
        return None, None, None

def main():
    # Load model
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
        help="Upload an image to detect vertical strips"
    )
    
    if uploaded_file is not None:
        # Display original image
        image = Image.open(uploaded_file)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📸 Original Image")
            st.image(image, caption="Uploaded Image", use_container_width=True)
        
        with col2:
            st.subheader("📏 VStrip Detection Results")
            
            # Process image with vstrip.pt
            with st.spinner("Running vstrip.pt inference..."):
                det, annotated, vstrip_results = detect_vstrips(
                    image, vstrip_model, conf_threshold, iou_threshold, img_size, device
                )
            
            if det is not None:
                # Display vstrip results
                st.write(f"**Found {len(det)} vertical strip(s):**")
                
                # Convert BGR to RGB for display
                annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                st.image(annotated_rgb, caption="VStrip Detections", use_container_width=True)
                
                # Show vstrip details
                st.subheader("📋 Detection Details")
                vstrip_data = []
                for i in range(len(det)):
                    v_conf = det.confidence[i]
                    v_class_id = int(det.class_id[i])
                    v_x1, v_y1, v_x2, v_y2 = det.xyxy[i]
                    
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
                        "Y2": f"{v_y2:.1f}",
                        "Width": f"{v_x2 - v_x1:.1f}",
                        "Height": f"{v_y2 - v_y1:.1f}"
                    })
                
                if vstrip_data:
                    st.dataframe(vstrip_data, use_container_width=True)
                    
                    # Summary statistics
                    st.subheader("📊 Summary")
                    st.write(f"**Total Strips Detected:** {len(det)}")
                    if len(det) > 0:
                        avg_confidence = np.mean(det.confidence)
                        st.write(f"**Average Confidence:** {avg_confidence:.3f}")
                        
                        # Show confidence distribution
                        conf_values = det.confidence
                        st.write(f"**Confidence Range:** {np.min(conf_values):.3f} - {np.max(conf_values):.3f}")
            else:
                st.info("No vertical strips detected in this image.")
                st.write("Try adjusting the confidence threshold or uploading a different image.")
    
    # Instructions
    if uploaded_file is None:
        st.info("👆 Please upload an image to start vertical strip detection.")
        
        st.subheader("ℹ️ How to use:")
        st.markdown("""
        1. **Configure Model**: Set the path to your vstrip.pt model weights in the sidebar
        2. **Adjust Parameters**: Fine-tune confidence and IoU thresholds as needed
        3. **Upload Image**: Click "Browse files" to upload an image for detection
        4. **View Results**: 
           - VStrip.pt detects vertical strips in the image
           - View annotated image with detected strips
           - See detailed information in the table
        5. **Analyze**: Check confidence scores and strip dimensions
        """)

if __name__ == "__main__":
    main()
