#%% Superres.py

import os
import cv2
import numpy as np
from skimage.io import imread
from skimage.color import rgb2gray
from skimage.transform import radon, rescale
from scipy.ndimage import rotate
from scipy.signal import correlate2d

# def align_images(image, reference):
#     # Convert images to grayscale
#     image_gray = rgb2gray(image)
#     reference_gray = rgb2gray(reference)

#     # Compute the Radon transform for the image and reference
#     theta = np.linspace(0., 180., max(image_gray.shape), endpoint=False)
#     sinogram_image = radon(image_gray, theta=theta)
#     sinogram_reference = radon(reference_gray, theta=theta)

#     # Compute the cross-correlation between the two sinograms
#     correlation = correlate2d(sinogram_image, sinogram_reference)
#     angle = np.unravel_index(np.argmax(correlation), correlation.shape)[1]

#     # Rotate the image to align with the reference
#     aligned_image = rotate(image, angle)

#     return aligned_image



# def align_images_SURF(image, reference):
#     # Convert images to grayscale
#     image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     reference_gray = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)

#     # Create a SURF object
#     surf = cv2.xfeatures2d.SURF_create()

#     # Detect and compute SURF features for both images
#     kp1, des1 = surf.detectAndCompute(image_gray, None)
#     kp2, des2 = surf.detectAndCompute(reference_gray, None)

#     # Create a BFMatcher object
#     bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

#     # Match the SURF features
#     matches = bf.match(des1, des2)

#     # Sort the matches by distance
#     matches = sorted(matches, key=lambda x: x.distance)

#     # Extract the matched keypoints
#     src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
#     dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

#     # Compute the homography matrix
#     M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC)

#     # Get the size of the reference image
#     h, w = reference.shape[:2]

#     # Warp the image to align with the reference
#     aligned_image = cv2.warpPerspective(image, M, (w, h))

#     return aligned_image



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




# def add_padding(image, padding_size):
#     # Get the shape of the image
#     rows, cols = image.shape[:2]

#     # Check if the image is grayscale or color
#     if len(image.shape) == 2:
#         # Grayscale image
#         # Create a new image with the desired padding
#         padded_image = np.zeros((rows + 2 * padding_size, cols + 2 * padding_size))
#     else:
#         # Color image
#         # Get the number of color channels
#         channels = image.shape[2]

#         # Create a new image with the desired padding
#         padded_image = np.zeros((rows + 2 * padding_size, cols + 2 * padding_size, channels))

#     # Insert the original image into the padded image
#     padded_image[padding_size:padding_size + rows, padding_size:padding_size + cols] = image

#     return padded_image



# def remove_padding(padded_image, padding_size):
#     # Get the shape of the padded image
#     rows, cols = padded_image.shape[:2]

#     # Compute the size of the original image
#     original_rows = rows - 2 * padding_size
#     original_cols = cols - 2 * padding_size

#     # Check if the image is grayscale or color
#     if len(padded_image.shape) == 2:
#         # Grayscale image
#         # Extract the original image from the padded image
#         original_image = padded_image[padding_size:padding_size + original_rows, padding_size:padding_size + original_cols]
#     else:
#         # Color image
#         # Extract the original image from the padded image
#         original_image = padded_image[padding_size:padding_size + original_rows, padding_size:padding_size + original_cols, :]

#     return original_image




def superresolution(folder_path):
    # Import all .jpg images in the folder
    print('Importing images!')
    images = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".jpg"):
            img = cv2.imread(os.path.join(folder_path, filename))
            images.append(img)


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
        # warp_matrix = np.eye(2, 3, dtype=np.float32)
        # criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10000, 1e-03)
        # (cc, warp_matrix) = cv2.findTransformECC(cv2.cvtColor(reference_image,cv2.COLOR_BGR2GRAY), cv2.cvtColor(img,cv2.COLOR_BGR2GRAY), warp_matrix, cv2.MOTION_EUCLIDEAN, criteria)
        # aligned_img = cv2.warpAffine(img, warp_matrix, (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
        aligned_img = align_images_ORB(img, reference_image)
        aligned_images.append(aligned_img)
    print('Alignment finished!')
    
    # Average the stack of images using the median method
    print('Getting the median image...')
    median_image = np.median(np.array(aligned_images), axis=0).astype(np.uint8)

    # Apply a 200% sharpening filter to the averaged image with a 2 pixel radius
    # kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])/sharp
    # sharpened_image = cv2.filter2D(median_image, -1, kernel)
    # kernel = np.array([[0,-1,0], [-1,5*sharp/2,-1], [0,-1,0]])/sharp
    # sharpened_image = cv2.filter2D(median_image, -1, kernel)

    # b, g, r = cv2.split(median_image)

    # # Apply a Gaussian blur to each color channel
    # blurred_b = cv2.GaussianBlur(b, (0, 0), 3)
    # blurred_g = cv2.GaussianBlur(g, (0, 0), 3)
    # blurred_r = cv2.GaussianBlur(r, (0, 0), 3)

    # # Create the masks by subtracting the blurred channels from the original channels
    # mask_b = cv2.addWeighted(b, 1.5, blurred_b, -0.5, 0)
    # mask_g = cv2.addWeighted(g, 1.5, blurred_g, -0.5, 0)
    # mask_r = cv2.addWeighted(r, 1.5, blurred_r, -0.5, 0)

    # # Add the masks back to the original channels
    # sharpened_b = cv2.addWeighted(b, 1, mask_b, 1, 0)
    # sharpened_g = cv2.addWeighted(g, 1, mask_g, 1, 0)
    # sharpened_r = cv2.addWeighted(r, 1, mask_r, 1, 0)

    # # Merge the sharpened color channels back into a single image
    # sharpened = cv2.merge((sharpened_b, sharpened_g, sharpened_r))

    # Export the sharpened image to a .png named "SUPERRES.png" in the same folder the image stack was imported from
    # cv2.imwrite(os.path.join(folder_path, "SUPERRES_med_sharp.png"), sharpened_image)
    print('Saving median image...')
    cv2.imwrite(os.path.join(folder_path, "SUPERRES_med.png"), median_image)
    # cv2.imwrite(os.path.join(folder_path, "SUPERRES_sharp.png"), sharpened)
    print('Ready for sharpness/brightness adjustment! Close the tkinter window to cancel.')



from tkinter import *
from PIL import Image, ImageEnhance, ImageTk

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




# %%
folder_path = r'C:\Users\duncanla\Downloads\test_stack'
superresolution(folder_path)
GUI(folder_path)

# %%
