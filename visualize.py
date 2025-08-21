# importing packages required to sucessfully run the notebook

import cv2 # openCV -- contains several useful functions for image analysis and band identification
import skimage 
import numpy as np

import imutils # may need to use "sudo pip3 install imutils" because pip install imutils does not work
from imutils import contours 

import matplotlib.pyplot as plt
from ocr.helpers import implt, resize, ratio

print("OpenCV: " + cv2.__version__) 
print("Numpy: " + np.__version__)

plt.rcParams['figure.figsize'] = (9.0, 9.0)




# read and plot original image
# image is expected to be in jpeg format because this is the starard format for android phones (and most mobile devices)

# read image using openCV
image_path = "sample4.png"
image_orig = cv2.imread(image_path)

target_to_detect = "CMV" # target should be either 'CMV', ' BKV', or 'CXCL9'  

image_orig_copy = image_orig.copy() # making copy of the original image

# display image using python function
implt(image_orig)


height, width = image_orig.shape[:2]
crop_height = int(height * 0.10)  # 10% of the height

# Crop the image: remove 10% from top and 10% from bottom
# This keeps the middle 80% of the image
image_cropped = image_orig[crop_height:height-crop_height, :]

# Update the working copy
image_orig_copy = image_cropped.copy()

# Display cropped image
implt(image_cropped)


# convert image from BGR to RGB to 8-bit Grayscale
# image_RBG = cv2.cvtColor(image_orig, cv2.COLOR_BGR2RGB)
# image_GRAY = cv2.cvtColor(image_RBG, cv2.COLOR_RGB2GRAY)
image_GRAY = cv2.cvtColor(image_orig, cv2.COLOR_BGR2GRAY)
implt(image_GRAY)

# cv2.imshow("Grayscale image", image_GRAY)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# invert image colors (i.e., obtain negative of the image)
image_invert = cv2.bitwise_not(image_GRAY)
implt(image_invert)

image_invert_orig = image_invert.copy() # saving original inverted image such that contour box is not saved over

# cv2.imshow("Inverted image", image_invert_jpg)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


# apply Gaussian blur (helps with minimizing the effect of noisy artifactual bright-spots)
image_blur = cv2.GaussianBlur(image_invert, (11, 11), 0)

implt(image_blur)



import numpy as np
# Apply blur
image_blur = image_blur
h, w = image_blur.shape[:2]

# Compute cropping boundaries
top = int(0.1 * h)
bottom = int(0.9 * h)

# Crop the image
cropped = image_blur[top:bottom, :]
image_blur=cropped

# Now `cropped` is the image with top & bottom 10% removed

# Use 95th percentile threshold (targets only the hottest regions)
thresh = np.percentile(image_blur, 90)
image_thresh = cv2.threshold(image_blur, thresh, 255, cv2.THRESH_BINARY)[1]
# Clean up with morphological operations
kernel = np.ones((3,3), np.uint8)
image_thresh = cv2.morphologyEx(image_thresh, cv2.MORPH_OPEN, kernel)  # Remove noise
image_thresh = cv2.morphologyEx(image_thresh, cv2.MORPH_CLOSE, kernel)  # Fill holes

# ADD THIS LINE: Remove vertical structures with horizontal morphological opening
horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 1))  # Wide horizontal kernel
image_thresh = cv2.morphologyEx(image_thresh, cv2.MORPH_OPEN, horizontal_kernel)

# Remove small components
num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(image_thresh)
min_area = 50 # Adjust based on expected band size
cleaned_image = np.zeros_like(image_thresh)
for i in range(1, num_labels):
    # ADD THIS CONDITION: Check aspect ratio to avoid vertical structures
    area = stats[i, cv2.CC_STAT_AREA]
    width = stats[i, cv2.CC_STAT_WIDTH] 
    height = stats[i, cv2.CC_STAT_HEIGHT]
    aspect_ratio = width / height if height > 0 else 0
    
    if (area >= min_area and aspect_ratio > 2.0):  # Only keep horizontal structures
        cleaned_image[labels == i] = 255
# Display result
plt.imshow(cleaned_image, cmap='gray')

# cv2.imshow("Gaussian blur image", image_blur_jpg)
# cv2.waitKey(0)

# cv2.destroyAllWindows()