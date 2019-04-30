from __future__ import print_function
import sys
import cv2
#import cv
import json
import numpy as np
from skytracker import SkyTracker # < ---- just a test
from skytracker.calibrate_camera import DeFisheye

class frame_by_frame_defisheyer:
    default_param = {
            'datetime_mask': {'x': 410, 'y': 20, 'w': 500, 'h': 40},
            'output_video_name': 'tracking_video.avi',
            'output_video_fps': 25.0,
            'calibration_checkerboard_internal_corners': (6,9),
            'defisheye_balance' : 1.0,
            'timestamp_mask': True}

    def __init__(self, input_video_name, checkerboard_path, param=default_param):
        self.input_video_name = input_video_name
        self.checkerboard_path = checkerboard_path
        self.param = self.default_param
        if param is not None:
            self.param.update(param)

    def apply_datetime_mask(self,img):
        x = self.param['datetime_mask']['x']
        y = self.param['datetime_mask']['y']
        w = self.param['datetime_mask']['w']
        h = self.param['datetime_mask']['h']
        img_masked = np.array(img)
        img_masked[y:y+h, x:x+w] = np.zeros([h,w,3])
        return img_masked

    def run(self):
        cap = cv2.VideoCapture(self.input_video_name)
        if self.param['perform_calibration']:
            defisheye = DeFisheye(checkerboard_num_of_internal_corners = self.param['calibration_checkerboard_internal_corners'],
                                balance = self.param['defisheye_balance'] )
        #Here I'm running the fisheye calibration on the checkerboard video for this camera
            calib_dim, K, D = defisheye.calibrate(checkerboard_path = self.checkerboard_path)
            print (calib_dim)
        # Output files
        vid = None

        frame_count = -1
        while True:
            print('frame count: {0}'.format(frame_count))
            # Get frame, mask
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1

            # if timestamp_mask:
            frame = self.apply_datetime_mask(frame)

            if self.param['perform_calibration']:
                #frame = defisheye.undistort(img = frame, DIM = (calib_dim[0],calib_dim[1]), K = K, D = D)
                frame = defisheye.undistort(img = frame, DIM = calib_dim, K = K, D = D)

            #Here, saving to the de-fisheyed video
                if frame_count == 0 and self.param['output_video_name'] is not None:
                    vid = cv2.VideoWriter(
                        self.param['output_video_name'],
                        0x00000021,    # hack for cv2.VideoWriter_fourcc(*'MP4V')
                        self.param['output_video_fps'],
                        (frame.shape[1], frame.shape[0]),
                        )
                if vid is not None:
                    vid.write(frame)
