import pyrealsense2 as rs
import numpy as np
import cv2
import logging

####################################################################################################################################
# Configure depth and color streams...
# ...from Camera 1
pipeline_1 = rs.pipeline()
config_1 = rs.config()

config_1.enable_device('f0231937')
config_1.enable_stream(rs.stream.depth, 1024, 768, rs.format.z16, 30)
config_1.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)

# ...from Camera 2
pipeline_2 = rs.pipeline()
config_2 = rs.config()
config_2.enable_device('f0264175')
config_2.enable_stream(rs.stream.depth, 1024, 768, rs.format.z16, 30)
config_2.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)


# ...from Camera 3
pipeline_3 = rs.pipeline()
config_3 = rs.config()
config_3.enable_device('f1181177')
config_3.enable_stream(rs.stream.depth, 1024, 768, rs.format.z16, 30) 
config_3.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
#######################################################################################################################################

# Start streaming from both cameras
pipeline_1.start(config_1)
pipeline_2.start(config_2)
pipeline_3.start(config_3)
try:
    while True:

        # Camera 1
        # Wait for a coherent pair of frames: depth and color
        frames_1 = pipeline_1.wait_for_frames()
        depth_frame_1 = frames_1.get_depth_frame()
        color_frame_1 = frames_1.get_color_frame()
        if not depth_frame_1 or not color_frame_1:
            continue
        # Convert images to numpy arrays
        depth_image_1 = np.asanyarray(depth_frame_1.get_data())
        color_image_1 = np.asanyarray(color_frame_1.get_data())
        # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
        depth_colormap_1 = cv2.applyColorMap(cv2.convertScaleAbs(depth_image_1, alpha=0.5), cv2.COLORMAP_JET)

        # Camera 2
        # Wait for a coherent pair of frames: depth and color
        frames_2 = pipeline_2.wait_for_frames()
        depth_frame_2 = frames_2.get_depth_frame()
        color_frame_2 = frames_2.get_color_frame()
        if not depth_frame_2 or not color_frame_2:
            continue
        # Convert images to numpy arrays
        depth_image_2 = np.asanyarray(depth_frame_2.get_data())
        color_image_2 = np.asanyarray(color_frame_2.get_data())
        # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
        depth_colormap_2 = cv2.applyColorMap(cv2.convertScaleAbs(depth_image_2, alpha=0.5), cv2.COLORMAP_JET)

        # Camera 3
        # Wait for a coherent pair of frames: depth and color
        frames_3 = pipeline_3.wait_for_frames()
        depth_frame_3 = frames_3.get_depth_frame()
        color_frame_3 = frames_3.get_color_frame()
        if not depth_frame_3 or not color_frame_3:
            continue
        # Convert images to numpy arrays
        depth_image_3 = np.asanyarray(depth_frame_3.get_data())
        color_image_3 = np.asanyarray(color_frame_3.get_data())
        # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
        depth_colormap_3 = cv2.applyColorMap(cv2.convertScaleAbs(depth_image_3, alpha=0.5), cv2.COLORMAP_JET)     
#########################################################################################################################################
        
        depth_colormap_dim1 = depth_colormap_1.shape
        color_colormap_dim1 = color_image_1.shape
        depth_colormap_dim2 = depth_colormap_2.shape
        color_colormap_dim2 = color_image_2.shape
        depth_colormap_dim3 = depth_colormap_3.shape
        color_colormap_dim3 = color_image_3.shape
        
        # If depth and color resolutions are different, resize color image to match depth image for display
        if depth_colormap_dim1 != color_colormap_dim1:
            resized_color_image = cv2.resize(color_image_1, dsize=(depth_colormap_dim1[1], depth_colormap_dim1[0]), interpolation=cv2.INTER_AREA)
            images1= resized_color_image
            images2= depth_colormap_1
            images1 = np.hstack((resized_color_image, depth_colormap_1))
        else: 
            images1= color_image_1
            images2= depth_colormap_1
            images1 = np.hstack((color_image_1, depth_colormap_1))

        if depth_colormap_dim2 != color_colormap_dim2:
            resized_color_image = cv2.resize(color_image_2, dsize=(depth_colormap_dim2[1], depth_colormap_dim2[0]), interpolation=cv2.INTER_AREA)
            images3= resized_color_image
            images4= depth_colormap_2
            images2 = np.hstack((resized_color_image, depth_colormap_2))
        else: 
            images3= color_image_2
            images4= depth_colormap_2
            images2 = np.hstack((color_image_2, depth_colormap_2))

        if depth_colormap_dim3 != color_colormap_dim3:
            resized_color_image = cv2.resize(color_image_3, dsize=(depth_colormap_dim3[1], depth_colormap_dim3[0]), interpolation=cv2.INTER_AREA)
            images5= resized_color_image
            images6= depth_colormap_3
            images3 = np.hstack((resized_color_image, depth_colormap_3))
        else: 
            images5= color_image_3
            images6= depth_colormap_3
            images3 = np.hstack((color_image_3, depth_colormap_3))


        # Show images from both cameras
        cv2.namedWindow('RealSense1', cv2.WINDOW_NORMAL)
        cv2.imshow('RealSense1', images1)
        cv2.namedWindow('RealSense2', cv2.WINDOW_NORMAL)
        cv2.imshow('RealSense2', images2)
        cv2.namedWindow('RealSense3', cv2.WINDOW_NORMAL)
        cv2.imshow('RealSense3', images3)
        cv2.waitKey(1)

        # Save images and depth maps from both cameras by pressing 'spacebar'
        ch = cv2.waitKey(1)
        if ch==32:
            cv2.imwrite("my_image_1.jpg",color_image_1)
            cv2.imwrite("my_depth_1.jpg",depth_colormap_1)
            cv2.imwrite("my_image_2.jpg",color_image_2)
            cv2.imwrite("my_depth_2.jpg",depth_colormap_2)
            print("Save")


finally:

    # Stop streaming
    pipeline_1.stop()
    pipeline_2.stop()