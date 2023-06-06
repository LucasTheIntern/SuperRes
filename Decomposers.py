# Decompose gifs or videos into individual frames for use in stack superresolution.

def gif_decomp(fpath): # for gifs...
    import imageio 
    gif = imageio.imread(fpath) 
    numframes = len(gif) 
    print('Saving gif frames...') 
    for i in range(numframes): 
        imageio.imsave(fpath[0:-4] + "\\decomp\\frame_%d.png" % i, gif[i])

def vid_decomp(fpath): # for various video file types (mp4, avi, mov)
    import cv2
    import os
    video = cv2.VideoCapture(fpath)
    frame_width = int(video.get(3))
    frame_height = int(video.get(4))
    output_dir = fpath[0:-4] + "\\decomp\\"
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    while True:
        ret, frame = video.read()
        if not ret:
            break
        cv2.imwrite(os.path.join(output_dir, str(frame_count) + ".png"), frame)
        frame_count += 1
    video.release()
    cv2.destroyAllWindows()