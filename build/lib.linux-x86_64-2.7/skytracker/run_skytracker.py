import sys
import os.path
import time
from skytracker import SkyTracker

videofile = sys.argv[1]
t = time.strftime("%Y%m%d_%H%M%S", time.localtime())
print 'videofile = {0}'.format(videofile)

basename = os.path.basename(videofile)
basename_noext, dummy = os.path.splitext(basename)

#output_video_name = 'tracking_{0}.mp4'.format(basename_noext,t)
#blob_file_name = 'blob_data_{0}.txt'.format(basename_noext,t)
output_video_name = 'tracking_'+basename_noext+'_'+t+'.mp4'
blob_file_name = 'blob_data_'+basename_noext+'_'+t+'.txt'

param = {
        'bg_window_size': 3,
        'fg_threshold': 5,
        'datetime_mask': {'x': 430, 'y': 15, 'w': 500, 'h': 40}, 
        'min_area': 3, 
        'max_area': 200000,
        'open_kernel_size': (3,3),
	'close_kernel_size': (15,15),
        'output_video_name': output_video_name,
        'output_video_fps': 25.0,
        'blob_file_name': blob_file_name,
        'show_dev_images' : False,
        'min_interblob_spacing' : 2
        }

tracker = SkyTracker(input_video_name=videofile, param=param)
tracker.run()

