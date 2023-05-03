# SuperRes
Get superresolution from a stack of images using cv2 and PIL libraries in Python

Pixel-shift photography is a method of obtaining images of far higher resolution than those of which the sensor is directly capable. 
In some high-end cameras, the sensor is subjected to a "jitter" that allows for multiple sub-pixel shifts in the image. The images acquired are then upscaled and composited in such a way as to increase the resolution of the final image.

A similar technique can be performed using a low-end camera. Intentionally allowing a handheld camera to move very slightly between shots yields a batch of images to which a superresolution algorithm may be applied. However, the shifts between these images are uncontrolled and are often larger than a single pixel.
There are many existing tutorials for doing this in Photoshop or similar programs. For example, the tutorial linked below was used in the development of the SuperRes scripts.
https://petapixel.com/2019/10/29/how-to-double-the-resolution-of-your-smartphone-photos/

SuperRes3.py corrects for these large shifts and rotations using the ORB (Oriented FAST and Rotated BRIEF) algorithm to align the images before upscaling and finding the median image. 
