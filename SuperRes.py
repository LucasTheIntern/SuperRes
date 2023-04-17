#%% Superres.py

# Gift from the shiny new version of Bing!

# Request:
# Write a python script that makes a superresolution image from a stack of images by doing the following: 
# Import all .jpg images in a folder (the folder path should be an input). 
# Upscale those images by 200% using the nearest neighbor method. 
# Align all of the images. 
# Average the stack of images using the median method. 
# Apply a 200% sharpening filter to the averaged image with a 2 pixel radius. 
# Finally, export the sharpened image to a .png named "SUPERRES.png" in the same folder the image stack was imported from.

# Bing's comments:
# Here is a Python script that should do what you’re asking for:
# ...
# You can use this function by calling superresolution(folder_path) where folder_path is the path to the folder containing your stack of .jpg images. 
# This script uses OpenCV library for image processing. 
# Make sure you have it installed before running this script.

import os
import cv2
import numpy as np

def superresolution(folder_path,sharp):
    # Import all .jpg images in the folder
    images = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".jpg"):
            img = cv2.imread(os.path.join(folder_path, filename))
            images.append(img)

    # Upscale the images by 200% using the nearest neighbor method
    upscaled_images = []
    for img in images:
        upscaled_img = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST)
        upscaled_images.append(upscaled_img)

    # Align all of the images
    aligned_images = []
    reference_image = upscaled_images[0]
    for img in upscaled_images:
        warp_matrix = np.eye(2, 3, dtype=np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10000, 1e-03)
        (cc, warp_matrix) = cv2.findTransformECC(cv2.cvtColor(reference_image,cv2.COLOR_BGR2GRAY), cv2.cvtColor(img,cv2.COLOR_BGR2GRAY), warp_matrix, cv2.MOTION_EUCLIDEAN, criteria)
        aligned_img = cv2.warpAffine(img, warp_matrix, (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
        aligned_images.append(aligned_img)

    # Average the stack of images using the median method
    median_image = np.median(np.array(aligned_images), axis=0).astype(np.uint8)

    # Apply a 200% sharpening filter to the averaged image with a 2 pixel radius
    # kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])/sharp
    # sharpened_image = cv2.filter2D(median_image, -1, kernel)
    # kernel = np.array([[0,-1,0], [-1,5*sharp/2,-1], [0,-1,0]])/sharp
    # sharpened_image = cv2.filter2D(median_image, -1, kernel)

    b, g, r = cv2.split(median_image)

    # Apply a Gaussian blur to each color channel
    blurred_b = cv2.GaussianBlur(b, (0, 0), 3)
    blurred_g = cv2.GaussianBlur(g, (0, 0), 3)
    blurred_r = cv2.GaussianBlur(r, (0, 0), 3)

    # Create the masks by subtracting the blurred channels from the original channels
    mask_b = cv2.addWeighted(b, 1.5, blurred_b, -0.5, 0)
    mask_g = cv2.addWeighted(g, 1.5, blurred_g, -0.5, 0)
    mask_r = cv2.addWeighted(r, 1.5, blurred_r, -0.5, 0)

    # Add the masks back to the original channels
    sharpened_b = cv2.addWeighted(b, 1, mask_b, 1, 0)
    sharpened_g = cv2.addWeighted(g, 1, mask_g, 1, 0)
    sharpened_r = cv2.addWeighted(r, 1, mask_r, 1, 0)

    # Merge the sharpened color channels back into a single image
    sharpened = cv2.merge((sharpened_b, sharpened_g, sharpened_r))

    # Export the sharpened image to a .png named "SUPERRES.png" in the same folder the image stack was imported from
    # cv2.imwrite(os.path.join(folder_path, "SUPERRES_med_sharp.png"), sharpened_image)
    cv2.imwrite(os.path.join(folder_path, "SUPERRES_med.png"), median_image)
    cv2.imwrite(os.path.join(folder_path, "SUPERRES_sharp.png"), sharpened)
# %%
