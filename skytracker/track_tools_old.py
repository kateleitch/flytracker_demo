from __future__ import print_function
import cv2
import sys
import copy
import math
import numpy
import skytracker
import matplotlib.pyplot as plt


class BlobMatcher:
    """
    Generates pairwise blob matchings from raw blob data.
    """

    default_param = {'max_blobs': 10, 'max_dist': 300}

    def __init__(self, param=default_param):
        self.param = param

    def run(self, blob_data):
        return self.get_match_list(blob_data)

    def get_match_list(self,blob_data):

        match_list = []

        for curr_item, next_item in zip(blob_data[0:-1],blob_data[1:]):

            curr_frame = curr_item['frame']
            next_frame = next_item['frame']
            next_blob_list = next_item['blobs']
            curr_blob_list = curr_item['blobs']

            num_to_check = len(curr_blob_list)
            blob_pair_list = []

            # If there aren't too many blobs - check all candidate pairs and select the best matches
            if num_to_check <= self.param['max_blobs']:

                # Create and sort list of candidate blob pairs
                candidate_pair_list = []
                for curr_blob in curr_blob_list:
                    for next_blob in next_blob_list:
                        distance = blob_distance(curr_blob, next_blob)
                        candidate_pair_list.append((distance, (curr_blob, next_blob)))
                candidate_pair_list.sort()

                # Check blob pairs until we have check all in curr_blob_list
                while num_to_check > 0:
                    if len(candidate_pair_list) == 0:
                        break
                    distance, blob_pair = candidate_pair_list.pop(0)
                    if distance <= self.param['max_dist']:
                        blob_pair_list.append(blob_pair)
                        # Remove 2nd blob - which was matched to 1st - from list of candidates
                        candidate_pair_list = [item for item in candidate_pair_list if item[1][1] != blob_pair[1]] 
                    # Remove 1st blob from list of candidates
                    candidate_pair_list = [item for item in candidate_pair_list if item[1][0] != blob_pair[0]]
                    num_to_check -= 1

            match_data = {'frame_pair': (curr_frame, next_frame), 'blob_pair_list': blob_pair_list}
            match_list.append(match_data)

        return match_list

                
class BlobStitcher:
    """
    Stitches together pairwise blob matches into trajectories.
    """

    def __init__(self):
        self.match_list_working = []

    def run(self, match_list):
        track_list = self.get_track_list(match_list)
        return track_list

    def get_track_list(self,match_list):
        """
        Returns list of all tracks. 
        """
        track_list = []
        self.match_list_working = copy.deepcopy(match_list)
        for index in range(len(self.match_list_working)-1): 
            frame_pair = self.match_list_working[index]['frame_pair']
            for blob_pair in self.match_list_working[index]['blob_pair_list']:
                track = self.get_track(frame_pair, blob_pair,index+1) 
                track_list.append(track)
        self.match_list_working = []
        return track_list

    def get_track(self, frame_pair, blob_pair, index):
        """
        Returns track for given frame_pair, blob_pair found by search forward through the 
        working list of all blob pair matches until no more matches are found. Blob pairs are 
        removed from the working list of pair matches as they are added to tracks.
        """
        track = [{'frame': frame_pair[0], 'blob': blob_pair[0]}]
        next_frame_pair = self.match_list_working[index]['frame_pair']
        for next_blob_pair in self.match_list_working[index]['blob_pair_list']:
            if next_blob_pair[0] == blob_pair[1]:
                track.extend(self.get_track(next_frame_pair, next_blob_pair, index+1))
                self.match_list_working[index]['blob_pair_list'].remove(next_blob_pair)
                break;
        if len(track) == 1:
            track.append({'frame': frame_pair[1], 'blob': blob_pair[1]})
        return track


class TrackVideoCreator:

    def __init__(self, video_file, track_list):

        self.video_file = video_file
        self.track_list = track_list
        self.param = {
                'circle_radius_margin': 5, 
                'circle_radius_min': 8,
                'point_radius': 2
                } 

    def run(self):

        cap = cv2.VideoCapture(self.video_file)
        number_of_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        start_frame = self.track_list[0][0]['frame']

        frame_to_tracks_dict = self.get_frame_to_tracks_dict(start_frame, number_of_frames-1)

        frame_number = start_frame

        while True:

            print('frame: {0}'.format(frame_number))

            cap.set(cv2.CAP_PROP_POS_FRAMES,frame_number)
            ret, frame = cap.read()
            if not ret:
                break
            
            tracks_in_frame = frame_to_tracks_dict[frame_number]

            if tracks_in_frame:
                for track in tracks_in_frame:

                    # Draw line segments track points which arent from the current frame number
                    for (item0, item1) in zip(track[:-1], track[1:]):
                        frame0 = item0['frame']
                        frame1 = item1['frame']
                        if frame0 == frame_number or frame1 == frame_number:
                            continue
                        x0 = int(item0['blob']['centroid_x'])
                        y0 = int(item0['blob']['centroid_y'])
                        x1 = int(item1['blob']['centroid_x'])
                        y1 = int(item1['blob']['centroid_y'])
                        cv2.line(frame, (x0, y0), (x1, y1), (0,0,255))
                        cv2.circle(frame, (x0, y0), self.param['point_radius'], (0,0,255), cv2.FILLED)
                        cv2.circle(frame, (x1, y1), self.param['point_radius'], (0,0,255), cv2.FILLED)

                    # Draw circle and line segments for point from current frame
                    for i, item in enumerate(track):
                        print(i)
                        print(item)
                        if item['frame'] == frame_number:
                            x = item['blob']['centroid_x']
                            y = item['blob']['centroid_y']
                            area = item['blob']['area']
                            radius = int(numpy.sqrt(area/numpy.pi) + self.param['circle_radius_margin'])
                            radius = max(radius, int(self.param['circle_radius_min']))
                            cv2.circle(frame,(int(x),int(y)), radius, (255,0,0))
                            if i != 0:
                                prev_item = track[i-1]
                                self.draw_partial_line_seg(frame, item['blob'], prev_item['blob'], radius)
                            if i !=  len(track)-1:
                                next_item = track[i+1]
                                self.draw_partial_line_seg(frame, item['blob'], next_item['blob'], radius)

            cv2.imshow('frame',frame)

            if 1:
                wait_key_val = cv2.waitKey(0) & 0xFF
                if wait_key_val == ord('q'):
                    break
                if wait_key_val == ord('f'):
                    frame_number += 1
                if wait_key_val == ord('b'):
                    frame_number -= 1
            else:
                wait_key_val = cv2.waitKey(1) & 0xFF
                if wait_key_val == ord('q'):
                    break
                frame_number += 1

        # Clean up
        cap.release()
        cv2.destroyAllWindows()

    def get_frame_to_tracks_dict(self, start_frame, end_frame):

        frame_to_tracks_dict = {}

        # Empty list until start_frame
        for i in range(start_frame):
            frame_to_tracks_dict[i] = []

        # Loop over frames and add all tracks which contains frame to list for 
        # that frame.
        for i in range(start_frame, end_frame+1):
            frame_to_tracks_dict[i] = []
            for track in self.track_list:
                if any([i==item['frame'] for item in track]):
                    frame_to_tracks_dict[i].append(track)

        return frame_to_tracks_dict

    def draw_partial_line_seg(self,frame, blob0, blob1, radius):
        x0 = blob0['centroid_x']
        y0 = blob0['centroid_y']
        x1 = blob1['centroid_x']
        y1 = blob1['centroid_y']

        dx = x1 - x0
        dy = y1 - y0
        vec_len = numpy.sqrt(dx**2 + dy**2)
        ux = dx/vec_len
        uy = dy/vec_len

        x_circ = int(x0 + radius*ux)
        y_circ = int(y0 + radius*uy)

        x1 = int(x1)
        y1 = int(y1)

        cv2.line(frame,(x_circ, y_circ), (x1, y1), (0,0,255))
        cv2.circle(frame, (x1, y1), self.param['point_radius'], (0,0,255))


def filter_outlying_segments(track_list, multiplier=1, use_mad=False, filter_floor_pix=50):
    
    new_track_list = []
    change_flag_list = []

    debug_track_list = []

    for i, track in enumerate(track_list):

        x_vals =[]
        y_vals =[]
        flagged_indices = []

        if len(track) <= 2:
            change_flag_list.append(False)
            new_track_list.append(track)
            continue

        x_vals = [item['blob']['centroid_x'] for item in track]
        y_vals = [item['blob']['centroid_y'] for item in track]

        diff_x = numpy.array(numpy.diff(x_vals))
        diff_y = numpy.array(numpy.diff(y_vals))

        step_array = (numpy.sqrt(diff_x**2 + diff_y**2))
        
        if use_mad:
            intrapair_mad = get_mad(step_array)
            intrapair_median = numpy.median(step_array)
        else:
            intrapair_step_sigma = numpy.std(step_array)
            intrapair_step_mean = numpy.mean(step_array)

        for index, value in enumerate(step_array):
            if use_mad:
                if numpy.abs(value - intrapair_median) > max(intrapair_mad*multiplier, filter_floor_pix):
                    flagged_indices.append(index+1)
            else:
                if numpy.abs(value - intrapair_step_mean) > max(intrapair_step_sigma*multiplier,filter_floor_pix):
                    flagged_indices.append(index+1)
                    print('i: {0}, avg: {1:1.2f}, std: {2:1.2f}, max: {3:1.2f}, {4}'.format(i,intrapair_step_mean, intrapair_step_sigma, step_array.max(), len(track)))

        if len(flagged_indices) == 0:
            change_flag_list.append(False)
            new_track_list.append(track)
            continue

        debug_track_list.append(track)
    
        flagged_indices.insert(0,0)
        flagged_indices.append(len(track))


        for n, m in zip(flagged_indices[:-1], flagged_indices[1:]):
            new_track = track[n:m]
            if len(new_track) > 1:
                new_track_list.append(new_track)
                change_flag_list.append(True)

    return new_track_list, change_flag_list, debug_track_list
        


# Utility functions
# ------------------------------------------------------------------------------

def get_mad(data):
    median = numpy.median(data)
    return numpy.median(numpy.abs(data - median))

def blob_position(blob):
    return blob['centroid_x'], blob['centroid_y']

def blob_distance(blob_0, blob_1):
    x0, y0 = blob_position(blob_0)
    x1, y1 = blob_position(blob_1)
    return math.sqrt((x0-x1)**2 + (y0-y1)**2)






