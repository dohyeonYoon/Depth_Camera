import pyrealsense2 as rs
import numpy as np
import cv2

from realsense_device_manager import DeviceManager

def visualise_measurements(frames_devices):
  
    for (device, frame) in frames_devices.items():
        color_image = np.asarray(frame[rs.stream.color].get_data())
        text_str = "device"
        cv2.putText(color_image, text_str, (50,50), cv2.FONT_HERSHEY_PLAIN, 2, (0,255,0) )
        # Visualise the results
        text_str = 'Color image from RealSense Device Nr: ' + "device"
        cv2.namedWindow(text_str)
        cv2.imshow(text_str, color_image)
        cv2.waitKey(1)


# Define some constants 
resolution_width = 1024 # pixels
resolution_height = 768 # pixels
frame_rate = 15  # fps
dispose_frames_for_stablisation = 30  # frames

try:
    # Enable the streams from all the intel realsense devices
    rs_config = rs.config()
    rs_config.enable_stream(rs.stream.depth, resolution_width, resolution_height, rs.format.z16, frame_rate)
    rs_config.enable_stream(rs.stream.infrared, 1, resolution_width, resolution_height, rs.format.y8, frame_rate)
    rs_config.enable_stream(rs.stream.color, resolution_width, resolution_height, rs.format.bgr8, frame_rate)

    # Use the device manager class to enable the devices and get the frames
    device_manager = DeviceManager(rs.context(), rs_config)
    device_manager.enable_all_devices()
    
    # Allow some frames for the auto-exposure controller to stablise
    while 1:
        frames = device_manager.poll_frames()
        visualise_measurements(frames)

except KeyboardInterrupt:
    print("The program was interupted by the user. Closing the program...")

finally:
    device_manager.disable_streams()
    cv2.destroyAllWindows()