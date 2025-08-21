import os
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image, ImageOps
import tempfile
import zipfile
import io
from pathlib import Path
import argparse

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
        print(f"Error in preprocessing: {str(e)}")
        return None

def process_image(image_path, model, conf_thresh=0.25, iou_thresh=0.45, img_sz=640, device_type="cpu", use_preprocessing=True):
    """Process a single image and return detections"""
    try:
        # Load image
        image = Image.open(image_path)
        
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

        # Clean up temporary files
        os.unlink(tmp_path)
        if use_preprocessing and 'preprocessed_path' in locals():
            os.unlink(preprocessed_path)

        if len(results) > 0:
            return results[0], image
        else:
            return None, image

    except Exception as e:
        print(f"Error processing {image_path}: {str(e)}")
        return None, None

def save_cropped_detections(results, original_image, output_folder, image_name, padding=10):
    """Save cropped detections for a single image"""
    if results is None:
        print(f"No results for {image_name}")
        return 0
    
    if not hasattr(results, 'boxes') or results.boxes is None:
        print(f"No detections found in {image_name}")
        return 0
    
    if len(results.boxes) == 0:
        print(f"No detections found in {image_name}")
        return 0
    
    # Convert PIL image to numpy array
    original_img_rgb = np.array(original_image)
    
    # Create subfolder for this image
    image_folder = os.path.join(output_folder, f"{Path(image_name).stem}_detections")
    os.makedirs(image_folder, exist_ok=True)
    
    detections_saved = 0
    
    for detection_idx in range(len(results.boxes)):
        # Get bounding box coordinates
        x1, y1, x2, y2 = results.boxes.xyxy[detection_idx].astype(int)
        
        # Add padding around the crop
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(original_img_rgb.shape[1], x2 + padding)
        y2 = min(original_img_rgb.shape[0], y2 + padding)
        
        # Crop the image
        cropped_img = original_img_rgb[y1:y2, x1:x2]
        
        # Get detection info
        confidence = results.boxes.conf[detection_idx]
        class_id = int(results.boxes.cls[detection_idx])
        
        # Get class name if available
        if hasattr(results, 'names') and class_id in results.names:
            class_name = results.names[class_id]
        else:
            class_name = f"Class {class_id}"
        
        # Save cropped image
        cropped_pil = Image.fromarray(cropped_img)
        filename = f"crop_{detection_idx + 1}_{class_name}_{confidence:.3f}.png"
        filepath = os.path.join(image_folder, filename)
        cropped_pil.save(filepath, format='PNG')
        
        detections_saved += 1
        print(f"  Saved: {filename}")
    
    return detections_saved

def main():
    parser = argparse.ArgumentParser(description='Batch process images with YOLOv8 OBB')
    parser.add_argument('--input_folder', type=str, 
                       default='/Users/apple/Downloads/Rapid Card Images/input_imgs',
                       help='Input folder containing images')
    parser.add_argument('--output_folder', type=str, 
                       default='cropped_detections_batch',
                       help='Output folder for cropped detections')
    parser.add_argument('--model_path', type=str, 
                       default='runs/obb/train/weights/best1.pt',
                       help='Path to YOLO model weights')
    parser.add_argument('--conf_thresh', type=float, default=0.25,
                       help='Confidence threshold')
    parser.add_argument('--iou_thresh', type=float, default=0.45,
                       help='IOU threshold')
    parser.add_argument('--img_sz', type=int, default=640,
                       help='Image size for inference')
    parser.add_argument('--device', type=str, default='cpu',
                       help='Device to use (cpu/cuda)')
    parser.add_argument('--no_preprocessing', action='store_true',
                       help='Disable image preprocessing')
    
    args = parser.parse_args()
    
    # Create output folder
    os.makedirs(args.output_folder, exist_ok=True)
    
    # Load model
    print(f"Loading model from: {args.model_path}")
    model = YOLO(args.model_path)
    
    # Get all image files
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    image_files = []
    
    for file in os.listdir(args.input_folder):
        if Path(file).suffix.lower() in image_extensions:
            image_files.append(file)
    
    print(f"Found {len(image_files)} images to process")
    
    total_detections = 0
    
    # Process each image
    for i, image_file in enumerate(image_files, 1):
        image_path = os.path.join(args.input_folder, image_file)
        print(f"\n[{i}/{len(image_files)}] Processing: {image_file}")
        
        # Process image
        results, original_image = process_image(
            image_path, 
            model, 
            args.conf_thresh, 
            args.iou_thresh, 
            args.img_sz, 
            args.device,
            not args.no_preprocessing
        )
        
        if results is not None and original_image is not None:
            # Save cropped detections
            detections_saved = save_cropped_detections(
                results, 
                original_image, 
                args.output_folder, 
                image_file
            )
            total_detections += detections_saved
            print(f"  Total detections in this image: {detections_saved}")
        else:
            print(f"  Failed to process {image_file}")
    
    print(f"\n🎉 Batch processing complete!")
    print(f"📁 Output folder: {args.output_folder}")
    print(f"📊 Total detections saved: {total_detections}")
    print(f"🖼️  Images processed: {len(image_files)}")

if __name__ == "__main__":
    main()
