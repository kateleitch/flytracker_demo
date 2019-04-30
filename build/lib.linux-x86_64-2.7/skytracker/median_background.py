from __future__ import print_function
import cv2
import numpy as np


class MedianBackground:

    def __init__(self,window_size=11, threshold=10):
        self.window_size = window_size
        self.threshold = threshold
        self.reset()

    def update(self,frame):
        self.frame_list.append(frame)
        if len(self.frame_list) >= self.window_size:
            self.ready = True
            self.frame_list.pop(0)

        frame_array = np.array(self.frame_list,dtype=np.uint8)
        self.background = np.median(frame_array,axis=0)
        self.background = np.array(self.background,dtype=np.uint8)

        # Get foreground  
        diff_frame = cv2.absdiff(frame,self.background)
        # NOTE: replace 255 with max value form dtype
        ret, self.foreground_mask = cv2.threshold(diff_frame, self.threshold, np.iinfo(frame.dtype).max, cv2.THRESH_BINARY)
        self.foreground = cv2.bitwise_and(frame, frame, mask=self.foreground_mask)

    def reset(self):
        self.ready = False
        self.foreground_mask = None
        self.foreground = None
        self.background = None 
        self.frame_list = []


