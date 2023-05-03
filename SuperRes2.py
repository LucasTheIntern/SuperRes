#%% Superres.py

import os
import cv2
import numpy as np

def superresolution(folder_path):
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
    cv2.imwrite(os.path.join(folder_path, "SUPERRES_med.png"), median_image)
    # cv2.imwrite(os.path.join(folder_path, "SUPERRES_sharp.png"), sharpened)




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
folder_path = r'C:\Users\fiddl\Downloads\test_stack_2'
superresolution(folder_path)
GUI(folder_path)

# %%
