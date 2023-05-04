# SuperRes
Get superresolution from a stack of images using cv2 and PIL libraries in Python
Some functions constructed with the assistance of Bing AI

Pixel-shift photography is a method of obtaining images of far higher resolution than those of which the sensor is directly capable with low noise. 
In some high-end cameras, the sensor is subjected to a "jitter" that allows for multiple sub-pixel shifts in the image. The images acquired are then upscaled and composited in such a way as to increase the resolution of the final image.

A similar technique can be performed using a low-end camera. Intentionally allowing a handheld camera to move very slightly between multiple shots yields a batch of images to which a superresolution algorithm may be applied. However, the shifts between these images are uncontrolled and are often larger than a single pixel.
There are many existing tutorials for doing this in Photoshop or similar programs. For example, the tutorial linked below was used in the development of the SuperRes scripts.
https://www.dpreview.com/articles/0727694641/here-s-how-to-pixel-shift-with-any-camera
Photos captured by smartphones in particular can see significant improvements by this method due to the fact that their small sensors result in noisy images. 

SuperRes.py corrects for "large" (>1 pixel) shifts and rotations using the ORB (Oriented FAST and Rotated BRIEF) algorithm to align the images before upscaling and finding the median image. However, large shifts are not reccomended, as the alignment algorithm can only do so much. Photos with moving subjects or significant changes in camera position will result in "ghosting" (blurry/smeared edges), especially for objects close to the camera. 

You will need: 
    A folder multiple photos all taken from the same location. File format: .jpg OR .png
    Python 3 and editor
    Python libraries:
        os
        cv2
        numpy
        tkinter
        PIL

.gif files are currently unsupported due to memory limitations... But there is a small portion of commented code for this, should you feel like working on that!

Based on articles I've read (have not tested for myself), 20 photos is a rough practical limit for this method. Beyond this, little to no improvement may be seen, and the program's run-time only increases. 

To use this script, supply a path (variable: folder_path at the top of the script) and run the code. The script will print statements to indicate progress. Once the median image has been obtained, a simple GUI window opens with sliders for controlling the brightness and sharpness of the image. If you desire to make these and other edits in a professional editor, you may simply close the GUI. The median image is saved to the same folder that was supplied by the user under the name "SUPERRES_med.png". If edited in the GUI, this image is also saved to that folder as "sharp_bright.png" once the "save" button has been pressed. 

Shown below is a zoomed-in comparison between the first image in a sequence (the image to which all remaining images are aligned) and the resulting superresolution image. This is a small drawer handle that was halfway across the room in low lighting conditions. Note the reduction in noise and smoothing of edges. Some slight ghosting is also visible, due to one image being taken nearly a centimeter from the main grouping. Try to avoid this when capturing your own images! 



