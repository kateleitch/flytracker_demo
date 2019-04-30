from __future__ import print_function
import cv2
assert cv2.__version__[0] == '3', 'The fisheye module requires opencv version >= 3.0.0'
import numpy as np
import os, sys

class DeFisheye:

    def __init__(self, checkerboard_num_of_internal_corners = (6,9), balance = 0.0):
        self.checkerboard_num_of_internal_corners = checkerboard_num_of_internal_corners
        self.balance = balance

    def get_filenames(self, path, contains, does_not_contain=['~', '.pyc']):
        cmd = 'ls ' + '"' + path + '"'
        ls = os.popen(cmd).read()
        all_filelist = ls.split('\n')
        try:
            all_filelist.remove('')
        except:
            pass
        filelist = []
        for i, filename in enumerate(all_filelist):
            if contains in filename:
                fileok = True
                for nc in does_not_contain:
                    if nc in filename:
                        fileok = False
                if fileok:
                    filelist.append( os.path.join(path, filename) )

        return filelist
        print (filelist)

    def calibrate(self, checkerboard_path):
        CHECKERBOARD = self.checkerboard_num_of_internal_corners

        subpix_criteria = (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001) #was .1

        ### KJL edit 11 May 2018 ####
        calibration_flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC+cv2.fisheye.CALIB_CHECK_COND+cv2.fisheye.CALIB_FIX_SKEW
        #calibration_flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC+cv2.fisheye.CALIB_FIX_SKEW
        #calibration_flags = cv2.fisheye.CALIB_CHECK_COND+cv2.fisheye.CALIB_FIX_SKEW

        objp = np.zeros((1, CHECKERBOARD[0]*CHECKERBOARD[1], 3), np.float32)
        objp[0,:,:2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)

        _img_shape = None
        objpoints = [] # 3d point in real world space
        imgpoints = [] # 2d points in image plane.

        #cap_checkerboard = cv2.VideoCapture(checkerboard_video_name)
        # calib_count = 0
        # while True:
        #     ret, frame = cap_checkerboard.read()
        #     calib_count += 1
        #     if calib_count%100 > 0: # this rejects 24 out of every 25 images; took video at 25 fps but couldn't move the checkerboard around that quickly obviously
        #         continue
        #     if not ret:
        #         break
        #     frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        path = checkerboard_path
        print (path)
        calib_frames = DeFisheye.get_filenames(self, path, contains='tl_000', does_not_contain=['~', '.pyc'])
        ##1 cap_checkerboard = cv2.VideoCapture(checkerboard_video_name)
        calib_count = 0
        ##2 while True:
        for frame_name in calib_frames:
            ##3 ret, frame = cap_checkerboard.read()
            #frame = load_image_from_filelist(frame_name)
            frame_color = cv2.imread(frame_name)
            calib_count += 1
            # if calib_count%2 > 0: # this accepts 1 out of int(frame_skips) images; took video at 25 fps but couldn't move the checkerboard around that quickly obviously
            #     continue
###4             if not ret:
#                 break
            frame = cv2.cvtColor(frame_color, cv2.COLOR_BGR2GRAY)
            frame_shape_for_calibrate = frame.shape[::-1]

            if _img_shape == None:
                _img_shape = frame.shape[:2]
            else:
                assert _img_shape == frame.shape[:2], "All images must share the same size."

            # Find the chess board corners
            flag, corners = cv2.findChessboardCorners(frame, CHECKERBOARD, cv2.CALIB_CB_ADAPTIVE_THRESH+cv2.CALIB_CB_FAST_CHECK+cv2.CALIB_CB_NORMALIZE_IMAGE)
            # If found, add object points, image points (after refining them)
            if flag == True:
                print ('corners found')
                objpoints.append(objp)
                cv2.cornerSubPix(frame,corners,(11,11),(-1,-1),subpix_criteria) #was (3,3), (-1, -1)
                imgpoints.append(corners)
                #print (corners)
                #print (corners.shape)
                # cv2.drawChessboardCorners(frame, (6,9), corners,ret)
                # cv2.imshow('img',frame)
                # cv2.waitKey(2000)
            else:
                print ('.............')
        #cv2.destroyAllWindows()

        N_OK = len(objpoints)
        K = np.zeros((3, 3))
        D = np.zeros((4, 1))
        rvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]
        tvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]
        rms, _, _, _, _ = \
            cv2.fisheye.calibrate(
                objpoints,
                imgpoints,
                frame_shape_for_calibrate, #KJL edit
                K,
                D,
                rvecs,
                tvecs,
                calibration_flags,
                (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6) #was 30, 1e-6
            )
        print (len(objpoints))
        print("Found " + str(N_OK) + " valid images for calibration")
        print("DIM=" + str(_img_shape[::-1]))
        print("K=np.array(" + str(K.tolist()) + ")")
        print("D=np.array(" + str(D.tolist()) + ")")

        return _img_shape[::-1], K, D

        #HERE I WANTO TO GENERATE AN X,Y PLOT OF RESIDUAL ERRORS TO SEE IF THERE ARE ANY SYSTEMATIC PROBLEMS E.G. WITH MY LENS MODEL
        #https://stackoverflow.com/questions/12794876/how-to-verify-the-correctness-of-calibration-of-a-webcam/12821056#12821056
        #cleanup
        cap_checkerboard.release()
    def undistort(self, img, DIM, K, D):
        balance = self.balance
        dim1 = img.shape[:2][::-1]  #... dim1 is the dimension of input image to un-distort
        dim2=None
        dim3=None

        assert dim1[0]/dim1[1] == DIM[0]/DIM[1], "Image to undistort needs to have same aspect ratio as the ones used in calibration"

        if not dim2:
            dim2 = dim1

        if not dim3:
            dim3 = dim1

        scaled_K = K * dim1[0] / DIM[0]  # The values of K is to scale with image dimension.
        scaled_K[2][2] = 1.0  # Except that K[2][2] is always 1.0

        # This is how scaled_K, dim2 and balance are used to determine the final K used to un-distort image. OpenCV document failed to make this clear!
        new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(scaled_K, D, dim2, np.eye(3), balance=balance)
        map1, map2 = cv2.fisheye.initUndistortRectifyMap(scaled_K, D, np.eye(3), new_K, dim3, cv2.CV_16SC2)
        undistorted_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        #print ('undistorted_img.shape: '+str(undistorted_img.shape))
        #print ('input image shape: ' +str(img.shape))
        return undistorted_img
        # path = './undistorted'
        # os.chdir(path)
        # cv2.imwrite(str(name), undistorted_img)
