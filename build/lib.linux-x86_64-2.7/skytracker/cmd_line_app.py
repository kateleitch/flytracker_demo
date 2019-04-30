from __future__ import print_function
import argparse
import json
from skytracker import SkyTracker


def app_main():

    parser = argparse.ArgumentParser(description='Demo application for tracking flies in upward facing cameras')
    parser.add_argument('videofile', help='video file for tracking')
    parser.add_argument('-c','--config', help='json configuration file')
    
    args = parser.parse_args()
    
    config_dict = None
    if args.config is not None:
        with open(args.config,'r') as f:
            config_dict = json.load(f)
    
    tracker = SkyTracker(input_video_name=args.videofile, param=config_dict)
    tracker.run()

