
from __future__ import print_function
import sys
import cv2
import numpy as np



class FrameStepper:

    def __init__(self, video_file):
        self.video_file = video_file


    def run(self):

        cap = cv2.VideoCapture(self.video_file)

        num_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) 

        frame_pos = 0 

        while True:

            print('frame_pos: {0}'.format(frame_pos))

            cap.set(cv2.CAP_PROP_POS_FRAMES,frame_pos)
            ret, frame = cap.read()
            if not ret:
                break

            cv2.imshow('frame',frame)

            wait_key_val = cv2.waitKey(0) & 0xFF
            if wait_key_val == ord('q'):
                break
            if wait_key_val == ord('f'):
                frame_pos += 1
            if wait_key_val == ord('b'):
                frame_pos -= 1
	    if wait_key_val == ord('j'):
		frame_pos = int(raw_input("Jump to frame: ")) 
            frame_pos = max(frame_pos,0)
            frame_pos = min(frame_pos,num_frames-1)



        # Clean up
        cap.release()
        cv2.destroyAllWindows()


# -------------------------------------------------------------------

if __name__ == '__main__':

    file_name = sys.argv[1]
    stepper = FrameStepper(file_name)
    stepper.run()
