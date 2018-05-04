from __future__ import print_function
import sys
import cv2
#import cv
import json
import numpy as np

from median_background import MedianBackground
from blob_finder import BlobFinder

### KJL 2018_04_18 #### trying to integrate de_fisheye with this workflow
from defisheye import DeFisheye # now need to make a difisheye.py script in this folder wiht a DeFisheye class in it


# DIM=(2592, 1944)
# K=np.array([[1262.179277672335, 0.0, 1275.466404337509], [0.0, 1261.8632092942269, 1001.5137484034931], [0.0, 0.0, 1.0]])
# D=np.array([[-0.03450569015228284], [0.003778400552623656], [-0.016075235821126792], [0.009150051563118222]])
# ########################

class SkyTracker:

    default_param = {
            'bg_window_size': 11,
            'fg_threshold': 10,
            'datetime_mask': {'x': 410, 'y': 20, 'w': 500, 'h': 40},
            'min_area': 0,
            'max_area': 100000,
            'open_kernel_size': (3,3),
            'close_kernel_size': (3,3),
            'kernel_shape': 'ellipse',
            'output_video_name': 'tracking_video.avi',
            'output_video_fps': 20.0,
            'blob_file_name': 'blob_data.json',
            'show_dev_images' : False,
            'min_interblob_spacing' : 2,
            'perform_calibration':True,
            'calibration_checkerboard_internal_corners': (6,9),
            'defisheye_balance' : 1.0}

    def __init__(self, input_video_name, checkerboard_video_name, param=default_param):
        self.input_video_name = input_video_name
        #KJL 2018_04_19
        self.checkerboard_video_name = checkerboard_video_name
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
        # KJL 2018_04_19


        bg_model = MedianBackground(
                window_size=self.param['bg_window_size'],
                threshold=self.param['fg_threshold']
                )

        blob_finder = BlobFinder(
                filter_by_area=True,
                min_area=self.param['min_area'],
                max_area=self.param['max_area'],
                open_kernel_size = self.param['open_kernel_size'],
                close_kernel_size = self.param['close_kernel_size'],
                kernel_shape = self.param['kernel_shape'],
		#---------KJL 2017_12_15
                min_interblob_spacing = self.param['min_interblob_spacing'])

        # KJL 2018_04_19
        if self.param['perform_calibration']:
            defisheye = DeFisheye(checkerboard_num_of_internal_corners = self.param['calibration_checkerboard_internal_corners'],
                                balance = self.param['defisheye_balance'] )
        #Here I'm running the fisheye calibration on the checkerboard video for this camera
            calib_dim, K, D = defisheye.calibrate(checkerboard_video_name = self.checkerboard_video_name)

        # Output files
        vid = None
        blob_fid = None

        if self.param['blob_file_name'] is not None:
            blob_fid = open(self.param['blob_file_name'], 'w')

        frame_count = -1

        while True:

            print('frame count: {0}'.format(frame_count))

            # Get frame, mask and convert to gray scale
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1

            frame = self.apply_datetime_mask(frame)
            ##### HERE IS WHERE I THINK defisheye.undistort COULD BE CALLED, ON A FRAME-BY-FRAME BASIS ####
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if self.param['perform_calibration']:
                frame = defisheye.undistort(img = frame, DIM = calib_dim, K = K, D = D)

            #############################################################

            if frame_count == 0 and self.param['output_video_name'] is not None:
                vid = cv2.VideoWriter(
                        self.param['output_video_name'],
                        0x00000021,    # hack for cv2.VideoWriter_fourcc(*'MP4V')
                        self.param['output_video_fps'],
                        (frame.shape[1], frame.shape[0]),
                        )
                #vid = cv2.VideoWriter(
                #        self.param['output_video_name'],
		#	cv.FOURCC('X','V','I','D'),
                #        self.param['output_video_fps'],
                #        (frame.shape[1], frame.shape[0]),
                #        )


            # Update background model
            bg_model.update(frame)
            if not bg_model.ready:
                continue

            # Find blobs and add data to blob file
            blob_list, blob_image, circ_image = blob_finder.find(frame,bg_model.foreground_mask)

            if vid is not None:
                vid.write(circ_image)

            if blob_fid is not None:
                frame_data = {'frame': frame_count, 'blobs' : blob_list}
                frame_data_json = json.dumps(frame_data)
                blob_fid.write('{0}\n'.format(frame_data_json))

            # Display preview images
            if self.param['show_dev_images']:
                cv2.imshow('original',frame)
                cv2.imshow('background', bg_model.background)
                cv2.imshow('foreground mask', bg_model.foreground_mask)
                cv2.imshow('blob_image', blob_image)
                cv2.imshow('circ_image', circ_image)
            else:
                cv2.imshow('circ_image', circ_image)

            wait_key_val = cv2.waitKey(1) & 0xFF
            if wait_key_val == ord('q'):
                break

        # Clean up
        cap.release()
        cv2.destroyAllWindows()

        if vid is not None:
            vid.release()

        if blob_fid is not None:
            blob_fid.close()



# ---------------------------------------------------------------------------------------

#if __name__ == '__main__':
#
#    input_video_name = sys.argv[1]
#
#    param = {
#            'bg_window_size': 11,
#            'fg_threshold': 10,
#            'datetime_mask': {'x': 430, 'y': 15, 'w': 500, 'h': 40},
#            'min_area': 1,
#            'max_area': 100000,
#            'open_kernel_size': (3,3),
#            }
#
#    tracker = SkyTracker(input_video_name=input_video_name, param=param)
#    tracker.run()
