#%% SuperRes.py
"""
Author: LucasTheIntern
Date: 5/4/2023
Purpose: Get a low-noise, superresolution photo from multiple lower quality photos.
"""

# folder_path = r'path_to_folder_containing_noisy_photos!'
# folder_path = r'C:\Users\duncanla\Downloads\goatstack'

import os
import cv2
import numpy as np
from tkinter import *
from PIL import Image, ImageEnhance, ImageTk
import time
from tkinter import messagebox



def align_images_ORB(image, reference):
    # Convert images to grayscale
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    reference_gray = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)

    # Create an ORB object
    orb = cv2.ORB_create()

    # Detect and compute ORB features for both images
    kp1, des1 = orb.detectAndCompute(image_gray, None)
    kp2, des2 = orb.detectAndCompute(reference_gray, None)

    # Create a BFMatcher object
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # Match the ORB features
    matches = bf.match(des1, des2)

    # Sort the matches by distance
    matches = sorted(matches, key=lambda x: x.distance)

    # Extract the matched keypoints
    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    # Compute the homography matrix
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC)

    # Get the size of the reference image
    h, w = reference.shape[:2]

    # Warp the image to align with the reference
    aligned_image = cv2.warpPerspective(image, M, (w, h))

    return aligned_image




def superresolution(folder_path, stop_flag_func):
    # Import all .jpg images in the folder
    print('Importing images!')
    images = []
    for filename in os.listdir(folder_path):
        if stop_flag_func():
            messagebox.showinfo("Script stopped", "The script has been stopped by the user.")
            return
        if filename.endswith(".jpg") or filename.endswith(".png"):
            img = cv2.imread(os.path.join(folder_path, filename))
            images.append(img)

    # Upscale the images by 200% using the nearest neighbor method
    print('Upscaling images...')
    upscaled_images = []
    for img in images:
        if stop_flag_func():
            messagebox.showinfo("Script stopped", "The script has been stopped by the user.")
            return
        upscaled_img = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST)
        upscaled_images.append(upscaled_img)

    # Align all of the images
    aligned_images = []
    reference_image = upscaled_images[0]
    k = 0
    for img in images:
        if stop_flag_func():
            messagebox.showinfo("Script stopped", "The script has been stopped by the user.")
            return
        k = k +1
        print('Aligning image ' + str(k) + ' of ' + str(len(upscaled_images)))
        aligned_img = align_images_ORB(img, reference_image)
        aligned_images.append(aligned_img)
    print('Alignment finished!')

    # Average the stack of images using the median method
    print('Getting the median image...')
    median_image = np.median(np.array(aligned_images), axis=0).astype(np.uint8)

    # Export the sharpened image to a .png named "SUPERRES.png" in the same folder the image stack was imported from
    print('Saving median image...')
    cv2.imwrite(os.path.join(folder_path, "SUPERRES_med.png"), median_image)
    print('Ready for sharpness/brightness adjustment! Close the tkinter window to cancel.')
