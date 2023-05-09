#%% SuperRes.py
"""
Author: LucasTheIntern
Date: 5/4/2023
Purpose: Get a low-noise, superresolution photo from multiple lower quality photos.
"""

folder_path = r'path_to_folder_containing_noisy_photos!'

import os
import cv2
import numpy as np
from tkinter import *
from PIL import Image, ImageEnhance, ImageTk



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




def superresolution(folder_path):
    # Import all .jpg images in the folder
    print('Importing images!')
    images = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            img = cv2.imread(os.path.join(folder_path, filename))
            images.append(img)
        # elif filename.endswith(".gif"):
        #     import imageio
        #     imageio.core.resource.get_remote_manager().request_kwargs['max_size'] = 512 * 1024 * 1024
        #     gif = imageio.mimread(os.path.join(folder_path, filename))
        #     images.extend(gif)


    # Upscale the images by 200% using the nearest neighbor method
    print('Upscaling images...')
    upscaled_images = []
    for img in images:
        upscaled_img = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST)
        upscaled_images.append(upscaled_img)
        
    # Align all of the images
    aligned_images = []
    reference_image = upscaled_images[0]
    k = 0
    for img in images:
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




def GUI(folder_path):
    # Set the path to the image
    image_path = folder_path + '\\SUPERRES_med.png'

    # Open the image
    image = Image.open(image_path)

    # Create the main window
    root = Tk()
    root.title('Image Editor')

    # Create a label to display the image
    image_label = Label(root)
    image_label.pack()

    # Function to update the image preview
    def update_image(sharpness, brightness):
        # Create a new image with the specified sharpness and brightness
        new_image = ImageEnhance.Sharpness(image).enhance(sharpness)
        new_image = ImageEnhance.Brightness(new_image).enhance(brightness)
        
        # Update the image label with the new image
        photo = ImageTk.PhotoImage(new_image)
        image_label.config(image=photo)
        image_label.image = photo

    # Function to save the image
    def save_image():
        # Get the current sharpness and brightness values
        sharpness = float(sharpness_scale.get())
        brightness = float(brightness_scale.get())
        
        # Create a new image with the specified sharpness and brightness
        new_image = ImageEnhance.Sharpness(image).enhance(sharpness)
        new_image = ImageEnhance.Brightness(new_image).enhance(brightness)
        
        # Ask the user where to save the image
        file_path = folder_path + '\\sharp_bright.png'
        
        # Save the image
        if file_path:
            new_image.save(file_path)

    # Create a scale to adjust the sharpness
    sharpness_scale = Scale(root, from_=0, to=2, resolution=0.01, length=400, orient=HORIZONTAL, label='Sharpness', command=lambda s: update_image(float(s), float(brightness_scale.get())))
    sharpness_scale.set(1)
    sharpness_scale.pack()

    # Create a scale to adjust the brightness
    brightness_scale = Scale(root, from_=0, to=2, resolution=0.01, length=400, orient=HORIZONTAL, label='Brightness', command=lambda s: update_image(float(sharpness_scale.get()), float(s)))
    brightness_scale.set(1)
    brightness_scale.pack()

    # Create a button to save the image
    save_button = Button(root, text='Save', command=save_image)
    save_button.pack()

    # Update the image preview initially
    update_image(1, 1)

    # Run the main loop
    root.mainloop()

    


superresolution(folder_path)
GUI(folder_path)



# %% Experimental: further increase resolution via LapSRN model in cv2:
# Requires OpenCV version 4.5.1+

# def AI_super_resolve(folder_path, scale):
#     # Set the path to the image
#     image_path = folder_path + '\\SUPERRES_med.png'

#     # Open the image
#     image = Image.open(image_path)

#     # Load the pre-trained LapSRN model
#     sr = cv2.dnn_superres.DnnSuperResImpl_create()
#     sr.readModel("LapSRN_x{}.pb".format(scale))
#     sr.setModel("lapsrn", scale)

#     # Upsample the image using the model
#     result = sr.upsample(image)

#     # Return the result
#     return result

# LapSRN = AI_super_resolve(folder_path, 8)

# %%
