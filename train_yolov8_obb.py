

!nvidia-smi

import os
HOME = os.getcwd()
print(HOME)

"""## Install YOLOv8

To install YOL0v8, run the following command:
"""

!pip install ultralytics -q

import ultralytics
ultralytics.checks()

"""Now, we can import YOLOv8 into our Notebook:"""

from ultralytics import YOLO

from IPython.display import display, Image



# Commented out IPython magic to ensure Python compatibility.
!mkdir {HOME}/datasets
# %cd {HOME}/datasets

!pip install roboflow --quiet

import roboflow

roboflow.login()

rf = roboflow.Roboflow()

project = rf.workspace("zenkardia").project("device_detect-hfhjx")
dataset = project.version(4).download("yolov8-obb")

import yaml

with open(f'{dataset.location}/data.yaml', 'r') as file:
    data = yaml.safe_load(file)

data['path'] = dataset.location

with open(f'{dataset.location}/data.yaml', 'w') as file:
    yaml.dump(data, file, sort_keys=False)

"""## Train a YOLOv8 OBB Object Detection Model

With our dataset downloaded, we can now train a YOLOv8 OBB object detection model. Run the code snippet below to start training your model:
"""

from ultralytics import YOLO

model = YOLO('yolov8n-obb.pt')

results = model.train(data=f"{dataset.location}/data.yaml", epochs=100, imgsz=640)

"""Your model will train for 100 epochs. After training, you can run test your model using an image from your test set.

## Test the OBB Object Detection Model

Let's test our OBB detection model on an image:
"""

model = YOLO('runs/obb/train2/weights/best.pt')

import os
import random

random_file = random.choice(os.listdir(f"{dataset.location}/test/images"))
file_name = os.path.join(f"{dataset.location}/test/images", random_file)

results = model(file_name)

print(results[0])

"""We can visualize our oriented bounding box predictions using the following code:"""

# !pip install supervision -q

import supervision as sv
import cv2

detections = sv.Detections.from_ultralytics(results[0])

oriented_box_annotator = sv.OrientedBoxAnnotator()
annotated_frame = oriented_box_annotator.annotate(
    scene=cv2.imread(file_name),
    detections=detections
)

sv.plot_image(image=annotated_frame, size=(16, 16))

# If not installed:
# !pip install ultralytics supervision -q

import os, random, cv2
from ultralytics import YOLO
import supervision as sv

# ---- Load model (trained OBB weights) ----
model = YOLO('runs/obb/train2/weights/best.pt')

# ---- Pick a random image from your test set ----
img_dir = f"{dataset.location}/test/images"   # assumes dataset.location is already defined
rand_name = random.choice(os.listdir(img_dir))
file_path = os.path.join(img_dir, rand_name)

# ---- Inference on CPU ----
results = model.predict(
    source=file_path,
    device="cpu",      # << force CPU
    half=False,        # keep FP32 on CPU
    imgsz=640,         # adjust if you trained at a different size
    conf=0.25,         # tweak threshold as needed
    iou=0.45,
    verbose=False
)

# ---- Visualize oriented boxes ----
det = sv.Detections.from_ultralytics(results[0])
annotator = sv.OrientedBoxAnnotator()

frame_bgr = cv2.imread(file_path)             # supervision expects BGR np.array; cv2.imread is fine
annotated = annotator.annotate(scene=frame_bgr, detections=det)

# Show (in notebook) and save
sv.plot_image(image=annotated, size=(16, 16))

out_dir = "inference_obb_out"
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, f"annotated_{rand_name}")
cv2.imwrite(out_path, annotated)
print(f"Saved: {out_path}")



