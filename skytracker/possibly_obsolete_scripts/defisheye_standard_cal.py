from __future__ import print_function
import cv2
#assert cv2.__version__[0] == '3', 'The fisheye module requires opencv version >= 3.0.0'
import numpy as np

class DeFisheye:

    def __init__(self, checkerboard_num_of_internal_corners = (6,9), balance = 0.0):
        self.checkerboard_num_of_internal_corners = checkerboard_num_of_internal_corners
        self.balance = balance

    def calibrate(self, checkerboard_video_name):
        CHECKERBOARD = self.checkerboard_num_of_internal_corners

        subpix_criteria = (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001) #was .1

        ### KJL edit 11 May 2018 ####
        calibration_flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC+cv2.fisheye.CALIB_CHECK_COND+cv2.fisheye.CALIB_FIX_SKEW
        #calibration_flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC+cv2.fisheye.CALIB_FIX_SKEW
        #calibration_flags = cv2.fisheye.CALIB_CHECK_COND+cv2.fisheye.CALIB_FIX_SKEW

        # objp = np.zeros((1, CHECKERBOARD[0]*CHECKERBOARD[1], 3), np.float32)
        # objp[0,:,:2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)

        objp = np.zeros((CHECKERBOARD[0]*CHECKERBOARD[1],3), np.float32)
        objp[:,:2] = np.mgrid[0:CHECKERBOARD[0],0:CHECKERBOARD[1]].T.reshape(-1,2)

        _img_shape = None
        objpoints = [] # 3d point in real world space
        imgpoints = [] # 2d points in image plane.

        cap_checkerboard = cv2.VideoCapture(checkerboard_video_name)
        calib_count = 0
        while True:
            ret, frame = cap_checkerboard.read()
            calib_count += 1
            if calib_count%202 > 0: # this rejects 24 out of every 25 images; took video at 25 fps but couldn't move the checkerboard around that quickly obviously
                continue
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame_shape_for_calibrate = frame.shape[::-1]

            if _img_shape == None:
                _img_shape = frame.shape[:2]
            else:
                assert _img_shape == frame.shape[:2], "All images must share the same size."

            # Find the chess board corners
            ##### flipped checkerboard dimensions!!!!! below ##
            flag, corners = cv2.findChessboardCorners(frame, CHECKERBOARD, cv2.CALIB_CB_ADAPTIVE_THRESH+cv2.CALIB_CB_FAST_CHECK+cv2.CALIB_CB_NORMALIZE_IMAGE)
            # If found, add object points, image points (after refining them)
            if flag == True:
                #print ('corners found')
                objpoints.append(objp)
                cv2.cornerSubPix(frame,corners,(11,11),(-1,-1),subpix_criteria) #was (3,3), (-1, -1) but this kept throwing an error
                imgpoints.append(corners)
                #print (corners)
                #print (corners.shape)
                cv2.drawChessboardCorners(frame, (6,9), corners,ret)
                cv2.imshow('img',frame)
                cv2.waitKey(2000)
            else:
                print ('.............')
        #cv2.destroyAllWindows()

        N_OK = len(objpoints)
        # K = np.zeros((3, 3))
        # D = np.zeros((4, 1))
        # rvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]
        # tvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]
        ret, K, D, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, frame_shape_for_calibrate,None,None)

        print("Found " + str(N_OK) + " valid images for calibration")
        # print("DIM=" + str(_img_shape[::-1]))
        # print("K=np.array(" + str(K.tolist()) + ")")
        # print("D=np.array(" + str(D.tolist()) + ")")
        print (K)
        print (D)
        cap_checkerboard.release()
        cv2.waitKey(20000)
        return _img_shape[::-1], K, D


        #HERE I WANTO TO GENERATE AN X,Y PLOT OF RESIDUAL ERRORS TO SEE IF THERE ARE ANY SYSTEMATIC PROBLEMS E.G. WITH MY LENS MODEL
        #https://stackoverflow.com/questions/12794876/how-to-verify-the-correctness-of-calibration-of-a-webcam/12821056#12821056
        #cleanup

    def undistort(self, img, K, D):
        balance = self.balance
        w,h = img.shape[:2]
        newCameraMtx, roi = cv2.getOptimalNewCameraMatrix(K, D, (w, h), balance)
        undistortedImg = cv2.undistort(img, K, D, None, newCameraMtx)
        return undistortedImg
