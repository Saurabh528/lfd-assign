import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import supervision as sv
from PIL import Image
import tempfile
import os
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import base64
import io
import json
from typing import Dict, Any
import openai

# Add OpenCV constants for connected components analysis
CC_STAT_AREA = cv2.CC_STAT_AREA
CC_STAT_WIDTH = cv2.CC_STAT_WIDTH
CC_STAT_HEIGHT = cv2.CC_STAT_HEIGHT

# OpenAI analysis functions
def _image_to_data_uri(image: Image.Image) -> str:
    """Convert PIL image to data URI for OpenAI API."""
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return f"data:image/png;base64,{img_str}"

def _build_user_prompt() -> str:
    """Build the user prompt for GPT-4o focused on accuracy."""
    return """You are analyzing a lateral flow device (LFD) image. Your task is to determine if control and test lines are present and assess their strength.

CRITICAL ANALYSIS REQUIREMENTS:

1. CONTROL LINE ANALYSIS:
   - Look for a control line (C) that indicates the test is working properly
   - Determine if present: YES or NO
   - If present, assess strength: STRONG or FAINT

2. TEST LINE ANALYSIS:
   - Look for test line(s) that indicate the presence of target substance
   - Determine if present: YES or NO
   - If present, count how many test lines you see
   - For each test line, assess strength: STRONG or FAINT

3. IMAGE QUALITY CONSIDERATIONS:
   - Images may be blurry, low quality, or have poor lighting
   - Be SMART and look VERY CAREFULLY at all details
   - Examine the image multiple times from different angles
   - Look for subtle lines, shadows, or faint markings
   - Consider that lines might be partially visible or smudged

4. AMBIGUITY HANDLING:
   - If you're unsure about a line, examine it more carefully
   - Look for any faint traces, partial lines, or unclear markings
   - Consider lighting variations and shadows that might hide lines
   - Be thorough in your examination - don't miss subtle details

5. STRENGTH ASSESSMENT (CRITICAL):
   - STRONG: Bold, thick, dark, prominent line that is clearly visible
   - FAINT: Thin, light, barely visible, weak line that is hard to see
   - N/A: When line is completely absent/not detected
   - You MUST classify EVERY detected line as either STRONG or FAINT
   - Do NOT use vague terms like "clear", "distinct", "visible" - only STRONG or FAINT
   - Control line: Always assess as STRONG or FAINT if present
   - Test lines: Assess each individual test line as STRONG or FAINT
   - Be strict about the distinction - if you can see it but it's weak = FAINT

6. FALSE POSITIVE AVOIDANCE:
   - Do NOT count shadows, smudges, or lighting artifacts as lines
   - Only count actual test/control lines
   - Be conservative but thorough

IMPORTANT: Return your analysis in this EXACT JSON format:
{
  "control_line": {
    "present": true/false,
    "strength": "STRONG" or "FAINT" or "N/A"
  },
  "test_lines": {
    "present": true/false,
    "count": 0 or number of lines found,
    "strengths": ["STRONG", "FAINT", etc.] or []
  },
  "result_classification": "POSITIVE" or "NEGATIVE" or "INVALID" or "AMBIGUOUS",
  "ambiguity_score": 0.0-1.0,
  "image_quality_notes": "Image Quality: [GOOD/FAIR/POOR] | Lighting: [GOOD/FAIR/POOR] | Blur: [NONE/MILD/SEVERE] | Notes: [brief technical observations only]"
}

CRITICAL INSTRUCTIONS:
- Look VERY CAREFULLY at the entire image
- Examine multiple times to catch subtle details
- Be thorough in your analysis - don't rush
- Consider that blurry or low-quality images may hide important details
- If uncertain, examine more carefully rather than giving up
- ALWAYS provide strength values (STRONG/FAINT/N/A) for all lines
- ALWAYS follow the exact image quality template format
- Be consistent in your assessment criteria
- STRENGTH ASSESSMENT IS MANDATORY: Every detected line MUST be classified as STRONG or FAINT
- Do NOT use descriptive text - only STRONG, FAINT, or N/A
- Be strict: if line is visible but weak = FAINT, if line is bold and prominent = STRONG"""

def analyze_with_openai(image: Image.Image, api_key: str, model: str = "gpt-4o") -> Dict[str, Any]:
    """Analyze image using OpenAI API focused on accuracy."""
    try:
        client = openai.OpenAI(api_key=api_key)
        
        # Convert image to data URI
        data_uri = _image_to_data_uri(image)
        
        # Build prompt focused on accuracy
        user_prompt = _build_user_prompt()
        
        # Call OpenAI API
        response = client.chat.completions.create(
            model=model,
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "system",
                    "content": "You are a medical image analysis expert specializing in lateral flow devices. Your primary goal is ACCURACY - analyze images carefully and return the most accurate result possible. Format is secondary to accuracy."
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_prompt},
                        {"type": "image_url", "image_url": {"url": data_uri}}
                    ]
                }
            ],
            max_tokens=1000,
            temperature=0.1  # Low temperature for consistent results
        )
        
        # Parse response
        content = response.choices[0].message.content
        if not content:
            raise ValueError("Empty response from OpenAI API")
        
        # Parse JSON response
        try:
            result = json.loads(content)
        except json.JSONDecodeError as e:
            st.error(f"Invalid JSON response: {e}")
            st.error(f"Raw response: {content}")
            raise ValueError("Invalid JSON response from API")
        
        return result
        
    except Exception as e:
        st.error(f"Error calling OpenAI API: {str(e)}")
        raise

def render_openai_analysis(analysis: Dict[str, Any]) -> None:
    """Render the OpenAI analysis results."""
    result = analysis.get("result_classification")
    control_data = analysis.get("control_line", {})
    test_data = analysis.get("test_lines", {})
    ambiguity = analysis.get("ambiguity_score")
    image_quality = analysis.get("image_quality_notes", "")

    st.write("**Analysis Results:**")
    
    # Control line details
    control_present = control_data.get("present", False)
    control_strength = control_data.get("strength", "N/A")
    
    # Test line details
    test_present = test_data.get("present", False)
    test_count = test_data.get("count", 0)
    test_strengths = test_data.get("strengths", [])
    
    # Create detailed metrics display
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if control_present:
            st.metric("Control Line", f"✅ Present ({control_strength})")
        else:
            st.metric("Control Line", f"❌ Absent ({control_strength})")
    
    with col2:
        if test_present:
            strength_text = ", ".join(test_strengths) if test_strengths else "Unknown"
            st.metric("Test Lines", f"✅ {test_count} line(s) - {strength_text}")
        else:
            st.metric("Test Lines", f"❌ Absent ({len(test_strengths)} lines)")
    
    with col3:
        # Color code the result
        result_color = {
            "POSITIVE": "🟢",
            "NEGATIVE": "🔵", 
            "INVALID": "🔴",
            "AMBIGUOUS": "🟡"
        }.get(result, "⚪")
        
        st.metric("Final Result", f"{result_color} {result}")
        # if ambiguity is not None:
        #     st.metric("Ambiguity Score", f"{float(ambiguity):.2f}")
    
    # Image quality notes
    # if image_quality:
    #     st.info(f"📸 **Image Quality Notes**: {image_quality}")
    
    # Overall interpretation
    if result == "POSITIVE":
        st.success("🎯 **POSITIVE**: Both control and test lines detected")
    elif result == "NEGATIVE":
        st.info("🎯 **NEGATIVE**: Only control line detected")
    elif result == "INVALID":
        st.error("🎯 **INVALID**: No control line detected - test cannot be trusted")
  
    
    # Detailed breakdown
    with st.expander("📋 **Detailed Analysis Breakdown**"):
        st.write("**Control Line Analysis:**")
        st.write(f"- Present: {'Yes' if control_present else 'No'}")
        st.write(f"- Strength: {control_strength}")
        
        st.write("**Test Lines Analysis:**")
        st.write(f"- Present: {'Yes' if test_present else 'No'}")
        st.write(f"- Count: {test_count}")
        st.write(f"- Strengths: {', '.join(test_strengths) if test_strengths else 'None detected'}")
        
        st.write("**Overall Assessment:**")
        st.write(f"- Classification: {result}")

# Set page configuration
st.set_page_config(
    page_title="YOLOv8 OBB Object Detection",
    page_icon="🎯",
    layout="wide"
)

# Title and description
st.title("🎯 AI-powered lateral flow analysis")
st.markdown("Upload an image of lateral flow device to analyze.")



# Sidebar for model configuration
st.sidebar.header("⚙️ Model Configuration")

# Model path input
model_path = 'best1.pt'

# Detection parameters
conf_threshold = st.sidebar.slider(
    "Confidence Threshold", 
    min_value=0.60, 
    max_value=1.0, 
    value=0.65, 
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

st.sidebar.divider()

# OpenAI configuration
#st.sidebar.header("🤖 OpenAI Analysis")

# API Key input
openai_api_key = st.secrets["OPENAI_API_KEY"]

# Model selection
openai_model = "gpt-4o"

# Enable OpenAI analysis toggle




@st.cache_resource
def load_model(model_path):
    """Load YOLOv8 model with caching"""
    try:
        model = YOLO(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

from PIL import ImageOps
import tempfile

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

def detect_lines_in_image(img_array):
    """Detect lines in an image using the improved algorithm"""
    try:
        # Convert to grayscale if needed
        if len(img_array.shape) == 3:
            img = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else:
            img = img_array.copy()
        
        # Step 1: Contrast enhance
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enh = clahe.apply(img)
        
        # Step 2: Crop strip region
        h, w = enh.shape
        crop_top, crop_bottom = int(h*0.25), int(h*0.85)
        strip = enh[crop_top:crop_bottom, :]
        
        # Step 3: Darkness profile
        row_means = np.mean(strip, axis=1)
        darkness = 255 - row_means
        
        # Step 4: Smooth
        window = 12
        smoothed = np.convolve(darkness, np.ones(window)/window, mode='same')
        
        # Step 5: Find peaks
        peaks, props = find_peaks(smoothed, distance=30, prominence=17)
        
        # Remove edge peaks
        valid = [p for p in peaks if 0.1*len(smoothed) < p < 0.85*len(smoothed)]
        
        lines, labels = [], []
        if valid:
            prominences = [props["prominences"][list(peaks).index(p)] for p in valid]
            # Sort by vertical position
            rows = sorted([(valid[i] + crop_top, prominences[i]) for i in range(len(valid))])
            
            if len(rows) >= 1:
                # Always treat the topmost as Control
                row_c, prom_c = rows[0]
                lines.append(row_c)
                labels.append(("Control (C)", prom_c))
                
            if len(rows) >= 2:
                # If a second peak exists below, treat as Test
                row_t, prom_t = rows[1]
                lines.append(row_t)
                labels.append(("Test (T)", prom_t))
        
        # Visualization
        vis = cv2.cvtColor(enh, cv2.COLOR_GRAY2BGR)
        for (lbl, _), y in zip(labels, lines):
            cv2.line(vis, (0, y), (w, y), (0, 0, 255), 2)
            cv2.putText(vis, lbl, (5, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        # Classification
        if not labels:
            result = "INVALID TEST (no control line detected)"
        elif len(labels) == 1:
            result = f"NEGATIVE TEST → {labels[0][0]} detected"
        else:
            result = "POSITIVE TEST → Control + Test lines detected"
        
        return result, vis, smoothed, peaks, [lbl for lbl, _ in labels], lines
        
    except Exception as e:
        st.error(f"Error in line detection: {str(e)}")
        return None, None, None, None, None, None



def process_image(image, model, conf_thresh, iou_thresh, img_sz, device_type, use_preprocessing=True):
    try:
        # Fix orientation, drop alpha
        image = ImageOps.exif_transpose(image).convert("RGB")

        # Save to tmp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tf:
            tmp_path = tf.name
            image.save(tmp_path, format="JPEG", quality=95)

        # Apply preprocessing if enabled
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

        # Run inference on the processed image
        results = model.predict(
            source=inference_path,
            device=device_type,
            imgsz=img_sz,
            conf=conf_thresh,
            iou=iou_thresh,
            half=False,
            verbose=False
        )

        detections = sv.Detections.from_ultralytics(results[0])

        img_bgr = cv2.imread(tmp_path)
        annotator = sv.OrientedBoxAnnotator(thickness=2)
        annotated_frame = annotator.annotate(scene=img_bgr.copy(), detections=detections)

        annotated_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)

        # Return preprocessed path for cropping
        return annotated_rgb, detections, results[0], tmp_path, inference_path

    except Exception as e:
        st.error(f"Error processing image: {str(e)}")
        return None, None, None, None, None


def main():
    # Load model
    if os.path.exists(model_path):
        model = load_model(model_path)
        if model is None:
            st.error("Failed to load model. Please check the model path.")
            return
        st.sidebar.success("✅ Model loaded successfully!")
    else:
        st.error(f"Model file not found at: {model_path}")
        st.info("Please make sure the model path is correct and the file exists.")
        return
    
    # File upload
    uploaded_file = st.file_uploader(
        "Choose an image file", 
        type=['png', 'jpg', 'jpeg', 'webp'],
        help="Upload an image to detect objects"
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
            
            # Process image
            with st.spinner("Running inference..."):
                annotated_img, detections, results, tmp_path, inference_path = process_image(
                    image, model, conf_threshold, iou_threshold, img_size, device, use_preprocessing
                )
            
            if annotated_img is not None:
                # Display annotated image
                st.image(annotated_img, caption="Detected Objects", use_container_width=True)
                
                # Display raw detection results
                st.subheader("📋 Raw Detection Results")
                if detections is not None and len(detections) > 0:
                    # Create a detailed table of all detections
                    detection_data = []
                    for i in range(len(detections)):
                        # Get detection info
                        confidence = detections.confidence[i]
                        class_id = int(detections.class_id[i])
                        x1, y1, x2, y2 = detections.xyxy[i]
                        
                        # Get class name if available
                        if hasattr(results, 'names') and class_id in results.names:
                            class_name = results.names[class_id]
                        else:
                            class_name = f"Class {class_id}"
                        
                        # Calculate bounding box dimensions
                        width = x2 - x1
                        height = y2 - y1
                        area = width * height
                        
                        detection_data.append({
                            "Detection #": i + 1,
                            "Class": class_name,
                            "Class ID": class_id,
                            "Confidence": f"{confidence:.3f}",
                            "X1": f"{x1:.1f}",
                            "Y1": f"{y1:.1f}",
                            "X2": f"{x2:.1f}",
                            "Y2": f"{y2:.1f}",
                            "Width": f"{width:.1f}",
                            "Height": f"{height:.1f}",
                            "Area": f"{area:.1f}"
                        })
                    
                    # Display the table
                    st.dataframe(detection_data, use_container_width=True)
                    
                    # Add export functionality
                    import pandas as pd
                    df = pd.DataFrame(detection_data)
                    csv = df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Detection Results (CSV)",
                        data=csv,
                        file_name=f"detection_results_{uploaded_file.name}.csv",
                        mime="text/csv"
                    )
                    
                    st.markdown("---")
                
                # Detection statistics
                st.subheader("📊 Detection Statistics")
                if detections is not None and len(detections) > 0:
                    num_detections = len(detections)
                    st.metric("Number of Detections", num_detections)
                    
                    # Class distribution
                    if hasattr(detections, 'class_id') and detections.class_id is not None:
                        class_names = []
                        if hasattr(results, 'names'):
                            class_names = [results.names[int(cls_id)] for cls_id in detections.class_id]
                        else:
                            class_names = [f"Class {int(cls_id)}" for cls_id in detections.class_id]
                        
                        # Count detections per class
                        from collections import Counter
                        class_counts = Counter(class_names)
                        
                        st.write("**Detections by Class:**")
                        for class_name, count in class_counts.items():
                            st.write(f"- {class_name}: {count}")
                    
                    # Confidence scores
                    if hasattr(detections, 'confidence') and detections.confidence is not None:
                        avg_conf = np.mean(detections.confidence)
                        max_conf = np.max(detections.confidence)
                        min_conf = np.min(detections.confidence)
                        
                        st.write("**Confidence Scores:**")
                        st.write(f"- Average: {avg_conf:.3f}")
                        st.write(f"- Maximum: {max_conf:.3f}")
                        st.write(f"- Minimum: {min_conf:.3f}")
                else:
                    st.info("No objects detected in the image.")
                
                # Display cropped detections
                if detections is not None and len(detections) > 0 and inference_path is not None:
                    st.subheader("🔍 Cropped Detections (from Preprocessed Image)")
                    
                    # Get the preprocessed image for cropping
                    if use_preprocessing:
                        # Use preprocessed image for cropping
                        preprocessed_img = cv2.imread(inference_path)
                        if preprocessed_img is not None:
                            crop_source_img = cv2.cvtColor(preprocessed_img, cv2.COLOR_BGR2RGB)
                        else:
                            # Fallback to original image
                            original_img = cv2.imread(tmp_path)
                            crop_source_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
                    else:
                        # Use original image for cropping
                        original_img = cv2.imread(tmp_path)
                        crop_source_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
                    
                    # Create columns for cropped images
                    num_detections = len(detections)
                    cols_per_row = 3
                    num_rows = (num_detections + cols_per_row - 1) // cols_per_row
                    
                    st.write(f"**Found {num_detections} detection(s):**")
                    
                    # Add a toggle to show/hide cropped images
                    show_crops = st.checkbox("Show individual cropped detections", value=True)
                    
                    if show_crops:
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
                                        x2 = min(crop_source_img.shape[1], x2 + padding)
                                        y2 = min(crop_source_img.shape[0], y2 + padding)
                                        
                                        # Crop the image
                                        cropped_img = crop_source_img[y1:y2, x1:x2]
                                        
                                        # Get detection info
                                        confidence = detections.confidence[detection_idx]
                                        class_id = int(detections.class_id[detection_idx])
                                        
                                        # Get class name if available
                                        if hasattr(results, 'names') and class_id in results.names:
                                            class_name = results.names[class_id]
                                        else:
                                            class_name = f"Class {class_id}"
                                        
                                        # Display cropped image with info
                                        st.image(cropped_img, caption=f"{class_name} ({confidence:.3f})", use_container_width=True)
                                        
                                        # OpenAI Analysis Integration
                                        if openai_api_key:
                                            st.write("**🤖 AI-Powered Lateral Flow Analysis:**")
                                            
                                            # Create analyze button for this crop
                                            if st.button(f"🤖 Analyze The Lines", key=f"openai_analyze_{detection_idx}"):
                                                with st.spinner(f"Analyzing the strip for control and test lines"):
                                                    try:
                                                        # Get original image crop for OpenAI analysis (same as line detection)
                                                        original_img = cv2.imread(tmp_path)
                                                        if original_img is not None:
                                                            original_img_rgb = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
                                                            # Crop from original image using same coordinates
                                                            original_crop = original_img_rgb[y1:y2, x1:x2]
                                                            
                                                            # Convert original crop to PIL Image
                                                            crop_pil = Image.fromarray(original_crop)
                                                            
                                                            # Run OpenAI analysis on original crop
                                                            analysis_result = analyze_with_openai(crop_pil, openai_api_key, openai_model)
                                                            
                                                            # Display results
                                                            render_openai_analysis(analysis_result)
                                                            
                                                            # Show raw JSON in expander for debugging
                                                            
                                                            
                                                        else:
                                                            st.error("Could not load original image for OpenAI analysis.")
                                                        
                                                    except Exception as e:
                                                        st.error(f"OpenAI analysis failed: {str(e)}")
                                                        st.info("Please check your API key and try again.")
                                        
                                        elif enable_openai_analysis and not openai_api_key:
                                            st.warning("🔑 **OpenAI API Key Required**: Please enter your API key in the sidebar to enable AI analysis.")
                                        
                                        # Add download button for individual crop
                                        cropped_pil = Image.fromarray(cropped_img)
                                        # with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as crop_tmp:
                                        #     cropped_pil.save(crop_tmp.name, format='PNG')
                                        #     with open(crop_tmp.name, 'rb') as crop_file:
                                        #         st.download_button(
                                        #             label=f"📥 {class_name}",
                                        #             data=crop_file.read(),
                                        #             file_name=f"crop_{detection_idx}_{class_name}_{confidence:.3f}.png",
                                        #             mime="image/png",
                                        #             key=f"crop_{detection_idx}"
                                        #         )
                                        
                                        # Add line detection button
                                        if st.button(f"🔍 Detect Heatmaps", key=f"detect_lines_{detection_idx}"):
                                            with st.spinner(f"Detecting lines in {class_name}..."):
                                                # Get original image crop for line detection
                                                original_img = cv2.imread(tmp_path)
                                                if original_img is not None:
                                                    original_img_rgb = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
                                                    # Crop from original image using same coordinates
                                                    original_crop = original_img_rgb[y1:y2, x1:x2]
                                                    
                                                    # We'll run line detection after vstrip detection, using vstrip crop
                                                    # For now, just prepare the original crop for later visualization
                                                    result, vis, smoothed, peaks, labels, lines = None, None, None, None, None, None
                                                    
                                                    # Apply vstrip.pt model to detect vertical strips
                                                    st.write("**🔍 Detecting Vertical Strips with vstrip.pt model:**")
                                                    try:
                                                        # Load vstrip model
                                                        vstrip_model = YOLO('vstrip.pt')
                                                        
                                                        # Save cropped image to temp file for inference
                                                        with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as temp_crop:
                                                            crop_pil = Image.fromarray(original_crop)
                                                            crop_pil.save(temp_crop.name, format='PNG')
                                                            
                                                            # Run vstrip detection
                                                            vstrip_results = vstrip_model.predict(
                                                                source=temp_crop.name,
                                                                device=device,
                                                                imgsz=640,
                                                                conf=0.25,
                                                                iou=0.45,
                                                                half=False,
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
                                                                vstrip_vis = original_crop.copy()
                                                                vstrip_annotator = sv.OrientedBoxAnnotator(thickness=2, color=sv.ColorPalette.DEFAULT)
                                                                vstrip_annotated = vstrip_annotator.annotate(scene=vstrip_vis, detections=vstrip_detections)
                                                                
                                                                # Display vstrip results
                                                                st.write(f"**Found {len(vstrip_detections)} vertical strip(s):**")
                                                                st.image(vstrip_annotated, caption="Vertical Strip Detections", use_container_width=True)
                                                                
                                                                # Show vstrip detection details
                                                                vstrip_data = []
                                                                for i in range(len(vstrip_detections)):
                                                                    v_conf = vstrip_detections.confidence[i]
                                                                    v_class_id = int(vstrip_detections.class_id[i])
                                                                    v_x1, v_y1, v_x2, v_y2 = vstrip_detections.xyxy[i]
                                                                    
                                                                    # Get class name if available
                                                                    if hasattr(vstrip_results[0], 'names') and v_class_id in vstrip_results[0].names:
                                                                        v_class_name = vstrip_results[0].names[v_class_id]
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
                                                                    st.write("**Vertical Strip Details:**")
                                                                    st.dataframe(vstrip_data, use_container_width=True)
                                                                    
                                                                    # Add heatmap visualization section
                                                                    st.write("**🔥 Heatmap Analysis of VStrip Crop:**")
                                                                    try:
                                                                        # Create a copy of the vstrip crop for processing
                                                                        vstrip_crop_copy = vstrip_crop.copy()
                                                                        
                                                                        # Convert to grayscale
                                                                        vstrip_gray = cv2.cvtColor(vstrip_crop_copy, cv2.COLOR_RGB2GRAY)
                                                                        
                                                                        # Invert image colors (negative)
                                                                        vstrip_invert = cv2.bitwise_not(vstrip_gray)
                                                                        
                                                                        # Apply Gaussian blur
                                                                        vstrip_blur = cv2.GaussianBlur(vstrip_invert, (11, 11), 0)
                                                                        
                                                                        # Crop top and bottom 10%
                                                                        h, w = vstrip_blur.shape[:2]
                                                                        top = int(0.01 * h)
                                                                        bottom = int(0.99 * h)
                                                                        vstrip_blur_cropped = vstrip_blur[top:bottom, :]
                                                                        
                                                                        # Apply 90th percentile threshold
                                                                        thresh = np.percentile(vstrip_blur_cropped, 92)
                                                                        vstrip_thresh = cv2.threshold(vstrip_blur, thresh, 255, cv2.THRESH_BINARY)[1]
                                                                        
                                                                        # Clean up with morphological operations
                                                                        kernel = np.ones((3,3), np.uint8)
                                                                        vstrip_thresh = cv2.morphologyEx(vstrip_thresh, cv2.MORPH_OPEN, kernel)
                                                                        vstrip_thresh = cv2.morphologyEx(vstrip_thresh, cv2.MORPH_CLOSE, kernel)
                                                                        
                                                                        # Remove vertical structures with horizontal morphological opening
                                                                        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 1))
                                                                        vstrip_thresh = cv2.morphologyEx(vstrip_thresh, cv2.MORPH_OPEN, horizontal_kernel)
                                                                        
                                                                        # Remove small components and keep only horizontal structures
                                                                        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(vstrip_thresh)
                                                                        min_area = 50
                                                                        cleaned_vstrip = np.zeros_like(vstrip_thresh)
                                                                        for i in range(1, num_labels):
                                                                            area = stats[i, CC_STAT_AREA]
                                                                            width = stats[i, CC_STAT_WIDTH] 
                                                                            height = stats[i, CC_STAT_HEIGHT]
                                                                            aspect_ratio = width / height if height > 0 else 0
                                                                            
                                                                            if (area >= min_area and aspect_ratio > 2.0):
                                                                                cleaned_vstrip[labels == i] = 255
                                                                        
                                                                        # Create heatmap visualizations with appropriate sizes
                                                                        st.write("**📊 Image Processing Pipeline Visualization:**")
                                                                        
                                                                        # Row 1: Original and Grayscale
                                                                        col1, col2 = st.columns(2)
                                                                        with col1:
                                                                            st.write("**Original VStrip Crop:**")
                                                                            st.image(vstrip_crop, caption="Original", use_container_width=True)
                                                                        with col2:
                                                                            st.write("**Grayscale Conversion:**")
                                                                            st.image(vstrip_gray, caption="Grayscale", use_container_width=True, channels="GRAY")
                                                                        
                                                                        # Row 2: Inverted and Blurred
                                                                        col3, col4 = st.columns(2)
                                                                        with col3:
                                                                            st.write("**Inverted Image:**")
                                                                            st.image(vstrip_invert, caption="Negative", use_container_width=True, channels="GRAY")
                                                                        with col4:
                                                                            st.write("**Gaussian Blur (11x11):**")
                                                                            st.image(vstrip_blur, caption="Blurred & Cropped", use_container_width=True, channels="GRAY")
                                                                        
                                                                        # Row 3: Threshold and Cleaned
                                                                        #col5, col6 = st.columns(2)
                                                                        # with col5:
                                                                        #     st.write("**90th Percentile Threshold:**")
                                                                        #     st.image(vstrip_thresh, caption="Threshold", use_container_width=True, channels="GRAY")
                                                                        # with col6:
                                                                        #     st.write("**Morphologically Cleaned:**")
                                                                        #     st.image(cleaned_vstrip, caption="Final Cleaned", use_container_width=True, channels="GRAY")
                                                                        
                                                                        # Row 4: Heatmap Analysis
                                                                        st.write("**🔥 Heatmap Analysis:**")
                                                                        col7, col8 = st.columns(2)
                                                                        with col7:
                                                                            # Create intensity heatmap
                                                                            fig_heatmap, ax_heatmap = plt.subplots(figsize=(6, 4))
                                                                            im = ax_heatmap.imshow(vstrip_blur_cropped, cmap='hot', aspect='auto')
                                                                            ax_heatmap.set_title('Intensity Heatmap')
                                                                            ax_heatmap.set_xlabel('Width (pixels)')
                                                                            ax_heatmap.set_ylabel('Height (pixels)')
                                                                            plt.colorbar(im, ax=ax_heatmap, label='Intensity')
                                                                            st.pyplot(fig_heatmap)
                                                                            plt.close()
                                                                        
                                                                        with col8:
                                                                            # Create histogram of intensities
                                                                            fig_hist, ax_hist = plt.subplots(figsize=(6, 4))
                                                                            ax_hist.hist(vstrip_blur_cropped.ravel(), bins=50, color='red', alpha=0.7)
                                                                            ax_hist.axvline(thresh, color='blue', linestyle='--', label=f'90th Percentile: {thresh:.1f}')
                                                                            ax_hist.set_title('Intensity Distribution')
                                                                            ax_hist.set_xlabel('Pixel Intensity')
                                                                            ax_hist.set_ylabel('Frequency')
                                                                            ax_hist.legend()
                                                                            ax_hist.grid(True, alpha=0.3)
                                                                            st.pyplot(fig_hist)
                                                                            plt.close()
                                                                        
                                                                        # Row 5: Statistical Analysis
                                                                        st.write("**📈 Statistical Analysis:**")
                                                                        col9, col10 = st.columns(2)
                                                                        with col9:
                                                                            # Row-wise intensity profile
                                                                            row_means = np.mean(vstrip_blur_cropped, axis=1)
                                                                            fig_profile, ax_profile = plt.subplots(figsize=(6, 4))
                                                                            ax_profile.plot(row_means, color='green', linewidth=2)
                                                                            ax_profile.set_title('Row-wise Intensity Profile')
                                                                            ax_profile.set_xlabel('Row Index')
                                                                            ax_profile.set_ylabel('Average Intensity')
                                                                            ax_profile.grid(True, alpha=0.3)
                                                                            st.pyplot(fig_profile)
                                                                            plt.close()
                                                                        
                                                                        with col10:
                                                                            # Component analysis
                                                                            if num_labels > 1:
                                                                                areas = [stats[i, CC_STAT_AREA] for i in range(1, num_labels)]
                                                                                widths = [stats[i, CC_STAT_WIDTH] for i in range(1, num_labels)]
                                                                                heights = [stats[i, CC_STAT_HEIGHT] for i in range(1, num_labels)]
                                                                                
                                                                                fig_comp, ax_comp = plt.subplots(figsize=(6, 4))
                                                                                ax_comp.scatter(widths, heights, s=areas, alpha=0.6, c=areas, cmap='viridis')
                                                                                ax_comp.set_title('Component Analysis')
                                                                                ax_comp.set_xlabel('Width (pixels)')
                                                                                ax_comp.set_ylabel('Height (pixels)')
                                                                                ax_comp.grid(True, alpha=0.3)
                                                                                st.pyplot(fig_comp)
                                                                                plt.close()
                                                                            else:
                                                                                st.info("No significant components detected for analysis")
                                                                        
                                                                        # Summary statistics
                                                                        # st.write("**📋 Processing Summary:**")
                                                                        # summary_data = {
                                                                        #     "Metric": [
                                                                        #         "Original Dimensions",
                                                                        #         "Cropped Dimensions", 
                                                                        #         "Threshold Value (90th %)",
                                                                        #         "Components Detected",
                                                                        #         "Min Area Filter",
                                                                        #         "Aspect Ratio Filter"
                                                                        #     ],
                                                                        #     "Value": [
                                                                        #         f"{vstrip_gray.shape[1]} × {vstrip_gray.shape[0]}",
                                                                        #         f"{vstrip_blur_cropped.shape[1]} × {vstrip_blur_cropped.shape[0]}",
                                                                        #         f"{thresh:.1f}",
                                                                        #         f"{num_labels - 1}",
                                                                        #         f"≥ {min_area} pixels",
                                                                        #         "> 2.0 (horizontal)"
                                                                        #     ]
                                                                        # }
                                                                        # st.dataframe(summary_data, use_container_width=True)
                                                                        
                                                                    except Exception as e:
                                                                        st.error(f"Error in heatmap visualization: {str(e)}")
                                                                        st.info("Heatmap analysis failed, but line detection will continue")
                                                                    
                                                                    # Now run line detection on the vstrip crop
                                                                    st.write("**🔍 Running Line Detection on VStrip Crop:**")
                                                                    try:
                                                                        # Get the vstrip crop for line detection
                                                                        if len(vstrip_detections) > 0:
                                                                            # Use the first vstrip detection for line detection
                                                                            v_x1, v_y1, v_x2, v_y2 = vstrip_detections.xyxy[0].astype(int)
                                                                            
                                                                            # Add padding around vstrip crop
                                                                            v_padding = 5
                                                                            v_x1 = max(0, v_x1 - v_padding)
                                                                            v_y1 = max(0, v_y1 - v_padding)
                                                                            v_x2 = min(original_crop.shape[1], v_x2 + v_padding)
                                                                            v_y2 = min(original_crop.shape[0], v_y2 + v_padding)
                                                                            
                                                                            # Crop the vstrip region from original crop
                                                                            vstrip_crop = original_crop[v_y1:v_y2, v_x1:v_x2]
                                                                            
                                                                            # Add heatmap visualization section
                                                                            st.write("**🔥 Heatmap Analysis of VStrip Crop:**")
                                                                            try:
                                                                                # Create a copy of the vstrip crop for processing
                                                                                vstrip_crop_copy = vstrip_crop.copy()
                                                                                
                                                                                # Convert to grayscale
                                                                                vstrip_gray = cv2.cvtColor(vstrip_crop_copy, cv2.COLOR_RGB2GRAY)
                                                                                
                                                                                # Invert image colors (negative)
                                                                                vstrip_invert = cv2.bitwise_not(vstrip_gray)
                                                                                
                                                                                # Apply Gaussian blur
                                                                                vstrip_blur = cv2.GaussianBlur(vstrip_invert, (11, 11), 0)
                                                                                
                                                                                # Crop top and bottom 10%
                                                                                h, w = vstrip_blur.shape[:2]
                                                                                top = int(0.1 * h)
                                                                                bottom = int(0.9 * h)
                                                                                vstrip_blur_cropped = vstrip_blur[top:bottom, :]
                                                                                
                                                                                # Apply 90th percentile threshold
                                                                                thresh = np.percentile(vstrip_blur_cropped, 90)
                                                                                vstrip_thresh = cv2.threshold(vstrip_blur_cropped, thresh, 255, cv2.THRESH_BINARY)[1]
                                                                                
                                                                                # Clean up with morphological operations
                                                                                kernel = np.ones((3,3), np.uint8)
                                                                                vstrip_thresh = cv2.morphologyEx(vstrip_thresh, cv2.MORPH_OPEN, kernel)
                                                                                vstrip_thresh = cv2.morphologyEx(vstrip_thresh, cv2.MORPH_CLOSE, kernel)
                                                                                
                                                                                # Remove vertical structures with horizontal morphological opening
                                                                                horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 1))
                                                                                vstrip_thresh = cv2.morphologyEx(vstrip_thresh, cv2.MORPH_OPEN, horizontal_kernel)
                                                                                
                                                                                # Remove small components and keep only horizontal structures
                                                                                num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(vstrip_thresh)
                                                                                min_area = 50
                                                                                cleaned_vstrip = np.zeros_like(vstrip_thresh)
                                                                                for i in range(1, num_labels):
                                                                                    area = stats[i, CC_STAT_AREA]
                                                                                    width = stats[i, CC_STAT_WIDTH] 
                                                                                    height = stats[i, CC_STAT_HEIGHT]
                                                                                    aspect_ratio = width / height if height > 0 else 0
                                                                                    
                                                                                    if (area >= min_area and aspect_ratio > 2.0):
                                                                                        cleaned_vstrip[labels == i] = 255
                                                                                
                                                                                # Create heatmap visualizations with appropriate sizes
                                                                                st.write("**📊 Image Processing Pipeline Visualization:**")
                                                                                
                                                                                # Row 1: Original and Grayscale
                                                                                col1, col2 = st.columns(2)
                                                                                with col1:
                                                                                    st.write("**Original VStrip Crop:**")
                                                                                    st.image(vstrip_crop, caption="Original", use_container_width=True)
                                                                                with col2:
                                                                                    st.write("**Grayscale Conversion:**")
                                                                                    st.image(vstrip_gray, caption="Grayscale", use_container_width=True, channels="GRAY")
                                                                                
                                                                                # Row 2: Inverted and Blurred
                                                                                col3, col4 = st.columns(2)
                                                                                with col3:
                                                                                    st.write("**Inverted Image:**")
                                                                                    st.image(vstrip_invert, caption="Negative", use_container_width=True, channels="GRAY")
                                                                                with col4:
                                                                                    st.write("**Gaussian Blur (11x11):**")
                                                                                    st.image(vstrip_blur_cropped, caption="Blurred & Cropped", use_container_width=True, channels="GRAY")
                                                                                
                                                                                # Row 3: Threshold and Cleaned
                                                                                col5, col6 = st.columns(2)
                                                                                # with col5:
                                                                                #     st.write("**90th Percentile Threshold:**")
                                                                                #     st.image(vstrip_thresh, caption="Threshold", use_container_width=True, channels="GRAY")
                                                                                # with col6:
                                                                                #     st.write("**Morphologically Cleaned:**")
                                                                                #     st.image(cleaned_vstrip, caption="Final Cleaned", use_container_width=True, channels="GRAY")
                                                                                
                                                                                # Row 4: Heatmap Analysis
                                                                                st.write("**🔥 Heatmap Analysis:**")
                                                                                col7, col8 = st.columns(2)
                                                                                with col7:
                                                                                    # Create intensity heatmap
                                                                                    fig_heatmap, ax_heatmap = plt.subplots(figsize=(6, 4))
                                                                                    im = ax_heatmap.imshow(vstrip_blur_cropped, cmap='hot', aspect='auto')
                                                                                    ax_heatmap.set_title('Intensity Heatmap')
                                                                                    ax_heatmap.set_xlabel('Width (pixels)')
                                                                                    ax_heatmap.set_ylabel('Height (pixels)')
                                                                                    plt.colorbar(im, ax=ax_heatmap, label='Intensity')
                                                                                    st.pyplot(fig_heatmap)
                                                                                    plt.close()
                                                                                
                                                                                with col8:
                                                                                    # Create histogram of intensities
                                                                                    fig_hist, ax_hist = plt.subplots(figsize=(6, 4))
                                                                                    ax_hist.hist(vstrip_blur_cropped.ravel(), bins=50, color='red', alpha=0.7)
                                                                                    ax_hist.axvline(thresh, color='blue', linestyle='--', label=f'90th Percentile: {thresh:.1f}')
                                                                                    ax_hist.set_title('Intensity Distribution')
                                                                                    ax_hist.set_xlabel('Pixel Intensity')
                                                                                    ax_hist.set_ylabel('Frequency')
                                                                                    ax_hist.legend()
                                                                                    ax_hist.grid(True, alpha=0.3)
                                                                                    st.pyplot(fig_hist)
                                                                                    plt.close()
                                                                                
                                                                                # Row 5: Statistical Analysis
                                                                                st.write("**📈 Statistical Analysis:**")
                                                                                col9, col10 = st.columns(2)
                                                                                with col9:
                                                                                    # Row-wise intensity profile
                                                                                    row_means = np.mean(vstrip_blur_cropped, axis=1)
                                                                                    fig_profile, ax_profile = plt.subplots(figsize=(6, 4))
                                                                                    ax_profile.plot(row_means, color='green', linewidth=2)
                                                                                    ax_profile.set_title('Row-wise Intensity Profile')
                                                                                    ax_profile.set_xlabel('Row Index')
                                                                                    ax_profile.set_ylabel('Average Intensity')
                                                                                    ax_profile.grid(True, alpha=0.3)
                                                                                    st.pyplot(fig_profile)
                                                                                    plt.close()
                                                                                
                                                                                with col10:
                                                                                    # Component analysis
                                                                                    if num_labels > 1:
                                                                                        areas = [stats[i, CC_STAT_AREA] for i in range(1, num_labels)]
                                                                                        widths = [stats[i, CC_STAT_WIDTH] for i in range(1, num_labels)]
                                                                                        heights = [stats[i, CC_STAT_HEIGHT] for i in range(1, num_labels)]
                                                                                        
                                                                                        fig_comp, ax_comp = plt.subplots(figsize=(6, 4))
                                                                                        ax_comp.scatter(widths, heights, s=areas, alpha=0.6, c=areas, cmap='viridis')
                                                                                        ax_comp.set_title('Component Analysis')
                                                                                        ax_comp.set_xlabel('Width (pixels)')
                                                                                        ax_comp.set_ylabel('Height (pixels)')
                                                                                        ax_comp.grid(True, alpha=0.3)
                                                                                        st.pyplot(fig_comp)
                                                                                        plt.close()
                                                                                    else:
                                                                                        st.info("No significant components detected for analysis")
                                                                                
                                                                                # Summary statistics
                                                                                # st.write("**📋 Processing Summary:**")
                                                                                # summary_data = {
                                                                                #     "Metric": [
                                                                                #         "Original Dimensions",
                                                                                #         "Cropped Dimensions", 
                                                                                #         "Threshold Value (90th %)",
                                                                                #         "Components Detected",
                                                                                #         "Min Area Filter",
                                                                                #         "Aspect Ratio Filter"
                                                                                #     ],
                                                                                #     "Value": [
                                                                                #         f"{vstrip_gray.shape[1]} × {vstrip_gray.shape[0]}",
                                                                                #         f"{vstrip_blur_cropped.shape[1]} × {vstrip_blur_cropped.shape[0]}",
                                                                                #         f"{thresh:.1f}",
                                                                                #         f"{num_labels - 1}",
                                                                                #         f"≥ {min_area} pixels",
                                                                                #         "> 2.0 (horizontal)"
                                                                                #     ]
                                                                                # }
                                                                                # st.dataframe(summary_data, use_container_width=True)
                                                                                
                                                                            except Exception as e:
                                                                                st.error(f"Error in heatmap visualization: {str(e)}")
                                                                                st.info("Heatmap analysis failed, but line detection will continue")
                                                                            
                                                                            # LINE DETECTION CODE REMOVED - KEPT IN MEMORY FOR FUTURE USE
                                                                            # LINE DETECTION CODE REMOVED - KEPT IN MEMORY FOR FUTURE USE
                                                                            # The following code has been commented out but preserved:
                                                                            # - Shadow removal preprocessing
                                                                            # - Line detection on preprocessed vstrip crop
                                                                            # - Line visualization on best1.pt crop
                                                                            # - Darkness profile plotting
                                                                            # - Line detection result display
                                                                            
                                                                            st.info("🔥 Heatmap analysis completed! Line detection has been temporarily disabled.")
                                                                        else:
                                                                            st.info("No vstrip detections available for line detection.")
                                                                    except Exception as e:
                                                                        st.error(f"Error in line detection: {str(e)}")
                                                            else:
                                                                st.info("No vertical strips detected in this crop.")
                                                                
                                                    except Exception as e:
                                                        st.error(f"Error in vertical strip detection: {str(e)}")
                                                    
                                                else:
                                                    st.error("Could not load original image for line detection.")
                                                    result, vis, smoothed, peaks, labels, lines = None, None, None, None, None, None
                                                
                                                # Line detection is now handled after vstrip detection
                                                # using vstrip crop for detection and best1.pt crop for visualization
                
                # Download cropped images section
                st.subheader("💾 Download Cropped Images")
                
                # Create a zip file with all cropped images
                import zipfile
                import io
                
                # Create zip file in memory
                zip_buffer = io.BytesIO()
                
                with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                    # Get original image for ZIP downloads
                    original_img = cv2.imread(tmp_path)
                    if original_img is not None:
                        original_img_rgb = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
                    else:
                        original_img_rgb = crop_source_img  # fallback
                    
                    # Add each cropped image to the zip
                    for detection_idx in range(len(detections)):
                        # Get bounding box coordinates
                        x1, y1, x2, y2 = detections.xyxy[detection_idx].astype(int)
                        
                        # Add padding around the crop
                        padding = 10
                        x1 = max(0, x1 - padding)
                        y1 = max(0, y1 - padding)
                        x2 = min(original_img_rgb.shape[1], x2 + padding)
                        y2 = min(original_img_rgb.shape[0], y2 + padding)
                        
                        # Crop from original image for ZIP download
                        cropped_img = original_img_rgb[y1:y2, x1:x2]
                        
                        # Get detection info
                        confidence = detections.confidence[detection_idx]
                        class_id = int(detections.class_id[detection_idx])
                        
                        # Get class name if available
                        if hasattr(results, 'names') and class_id in results.names:
                            class_name = results.names[class_id]
                        else:
                            class_name = f"Class {class_id}"
                        
                        # Convert to PIL and save to zip
                        cropped_pil = Image.fromarray(cropped_img)
                        img_buffer = io.BytesIO()
                        cropped_pil.save(img_buffer, format='PNG')
                        img_buffer.seek(0)
                        
                        # Add to zip with descriptive filename
                        filename = f"crop_{detection_idx + 1}_{class_name}_{confidence:.3f}.png"
                        zip_file.writestr(filename, img_buffer.getvalue())
                
                # Prepare zip for download
                zip_buffer.seek(0)
                
                # Download button for all cropped images
                st.download_button(
                    label="📥 Download All Cropped Images (ZIP)",
                    data=zip_buffer.getvalue(),
                    file_name=f"cropped_detections_{uploaded_file.name}.zip",
                    mime="application/zip"
                )
                
                st.info(f"📦 ZIP file contains {len(detections)} cropped images with detection details")
    
    # Instructions
    if uploaded_file is None:
        st.info("👆 Please upload an image to start object detection.")
        
        st.subheader("ℹ️ How to use:")
        st.markdown("""
        1. **Configure Model**: Set the path to your trained YOLOv8 OBB model weights in the sidebar
        2. **Adjust Parameters**: Fine-tune confidence and IoU thresholds as needed
        3. **Upload Image**: Click "Browse files" to upload an image for detection
        4. **View Results**: See the original and annotated images side by side
        5. **Download**: Save the annotated image with detected objects
        
        **Supported formats**: PNG, JPG, JPEG, WebP
        """)

if __name__ == "__main__":
    main()
