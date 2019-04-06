from __future__ import print_function
import cv2
import sys
import copy
import math
import numpy
import skytracker
import matplotlib.pyplot as plt
import json
from blob_data_tools import load_blob_data
import collections

class BlobMatcher:
    """
    Generates pairwise blob matchings from raw blob data.
    """
    default_param = {'max_blobs': 10, 'max_dist': 300} # max_dist refers to maximum distance, px, between successive blobs to be joined

    def __init__(self, param=default_param):
        self.param = param

    def run(self, blob_data):
        return self.get_match_list(blob_data)

    def get_match_list(self,blob_data):
        match_list = []
        for curr_item, next_item in zip(blob_data[0:-1],blob_data[1:]):
            #print (curr_item) #troubleshooting
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

                # Check blob pairs until we have checked all in curr_blob_list
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

    def __init__(self, video_file, track_list, shall_we_perform_ground_truthing, track_metadata_filename,existing_track_metadata_json):
        self.video_file = video_file
        self.track_list = track_list

        def convert_dict_from_unicode(data):
            if isinstance(data, basestring):
                return str(data)
            elif isinstance(data, collections.Mapping):
                return dict(map(convert_dict_from_unicode, data.iteritems()))
            elif isinstance(data, collections.Iterable):
                return type(data)(map(convert_dict_from_unicode, data))
            else:
                return data

        self.ground_truthing_annotation = shall_we_perform_ground_truthing
        self.track_metadata_filename = track_metadata_filename
        self.track_metadata_dict = {track_index:{} for track_index in range(len(self.track_list))}
        if existing_track_metadata_json != 'False':
            print (existing_track_metadata_json)
            d = load_blob_data(existing_track_metadata_json) # reads in existing metadata as a list.
            print (d)
            d = d[-1] #takes last dictionary in list (each entry was a distinct save point); d is now a dictionary in unicode
            for key_int in self.track_metadata_dict: # keys are type int
                key = str(key_int)
                if key in d:
                    if len(d[key]) != 0:
                        print ('POPULATING METADATA WITH OLD METADATA')
                        converted_from_unicode = dict([(int(k), str(v)) for k, v in d[key].items()])
                        self.track_metadata_dict[key_int] = converted_from_unicode
                        #self.track_metadata_dict[key_int] = convert_dict_from_unicode(d[key]) #populating self.track_metadata_dict with any pre-existing data from old file; allows me to pick up where I left off

        self.param = {
                'circle_radius_margin': 5,
                'circle_radius_min': 8,
                'point_radius': 2
                }
        self.metadata_flag_dict = {
                1 : 'track looks good',
                2 : 'track was split into 2 or more',
                3 : 'track flanked by false neg(s); track OK overall',
                4 : 'track flanked by false neg(s); current track does NOT represent overall trajectory',
                5 : 'false pos blob(s); track OK overall',
                6 : 'false pos blob(s); throw track out',
                7 : 'interesting behavior: sharp turn',
                8 : 'interesting behavior: strong sideslip'
                }

        self.cap = cv2.VideoCapture(self.video_file)
        self.number_of_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

        self.start_frame = self.track_list[0][0]['frame']
        self.frame_to_tracks_dict, self.frame_to_tracks_index_dict = self.get_frame_to_tracks_dict(self.start_frame, self.number_of_frames-1)

        self.mouse_event_flag = False
        self.mouse_event_data = {}

    def __del__(self):
        self.cap.release()

    def on_mouse_event(self,event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.mouse_event_flag = True
            self.mouse_event_data = {'x': x, 'y': y}
            print('enter (a) for add flag, (r) for remove flag, (i) ignore')

    def print_metadata_options(self, indent=2):
        indent_str = ' '*indent
        for k, v in self.metadata_flag_dict.iteritems():
            print('{}{} {}'.format(indent_str, k, v))

    def run(self):

        frame_number = self.start_frame

        cv2.namedWindow("frame")
        cv2.setMouseCallback("frame", self.on_mouse_event)

        if self.track_metadata_filename is not None:
            self.metadata_fid = open(self.track_metadata_filename, 'w')

        ###
        self.join_id = 0  #initializing this id, which will be used to label tracks-to-be-joined
        while True:

            print('frame: {0}'.format(frame_number))

            self.cap.set(cv2.CAP_PROP_POS_FRAMES,frame_number)
            ret, frame = self.cap.read()
            if not ret:
                break

            tracks_in_frame = self.frame_to_tracks_dict[frame_number] # this is a list of lists, most often length 1
            tracks_index_in_frame = self.frame_to_tracks_index_dict[frame_number]

            if tracks_in_frame:
                for track, track_index  in zip(tracks_in_frame, tracks_index_in_frame): #each track is a list of dictionaries [{'frame': xxxx, 'blob': {'min_x': 3.0, etc.}}, ... ]

                    print('  track: {}, flags: {}'.format(track_index, self.track_metadata_dict[track_index]))

                    trackcolor = (255,100,255)
                    if  self.track_metadata_dict[track_index]:
                        trackcolor = (0,0,255)

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

                        #if output_dictionary
                        cv2.line(frame, (x0, y0), (x1, y1), trackcolor)
                        cv2.circle(frame, (x0, y0), self.param['point_radius'], trackcolor, cv2.FILLED)
                        cv2.circle(frame, (x1, y1), self.param['point_radius'], trackcolor, cv2.FILLED)

                    # Draw circle and line segments for point from current frame
                    for i, item in enumerate(track):
                        if item['frame'] == frame_number:
                            x = item['blob']['centroid_x']
                            y = item['blob']['centroid_y']
                            area = item['blob']['area']
                            radius = int(numpy.sqrt(area/numpy.pi) + self.param['circle_radius_margin'])
                            radius = max(radius, int(self.param['circle_radius_min']))
                            cv2.circle(frame,(int(x),int(y)), radius, (255,0,0))
                            if i != 0:
                                prev_item = track[i-1]
                                self.draw_partial_line_seg(frame, item['blob'], prev_item['blob'], radius, trackcolor)
                            if i !=  len(track)-1:
                                next_item = track[i+1]
                                self.draw_partial_line_seg(frame, item['blob'], next_item['blob'], radius, trackcolor)

            cv2.imshow('frame',frame)

            frame_number = self.handle_main_waitkey(frame_number, tracks_in_frame, tracks_index_in_frame)
            if frame_number == -1:
                break

        if self.track_metadata_filename is not None:
            print ('now saving the metadata dictionary of track flags')
            metadata_json = json.dumps(self.track_metadata_dict)
            self.metadata_fid.write('{0}\n'.format(metadata_json))
        # Clean up
        cv2.destroyAllWindows()

    def handle_main_waitkey(self,frame_number,tracks_in_frame,tracks_index_in_frame):
        if 1:
            waitkey_val = cv2.waitKey(0) & 0xFF

            if self.mouse_event_flag:
                # On mouse event - option to set track flag

                event_point = self.mouse_event_data['x'], self.mouse_event_data['y']
                closest_track_index = get_closest_track_index(event_point, tracks_in_frame, tracks_index_in_frame)

                if waitkey_val == ord('a'):
                    print('add flag to track, pos = {}'.format(self.mouse_event_data))
                    self.print_metadata_options()
                    flag_key, flag_value = self.handle_flag_options_waitkey()
                    if flag_key is not None:
                        if closest_track_index is not None:
                            if flag_key == 2:
                                    print ('current join_id: ' + str(self.join_id))
                                    print ('enter "," to generate a new join_id for this track')
                                    print ('enter "." to use the current join_id for this track')
                                    waitkey_val = cv2.waitKey(0) & 0xFF
                                    if waitkey_val == ord(','):
                                        self.join_id += 1
                                        print ('generating new join_id')
                                        self.track_metadata_dict[closest_track_index][flag_key] = 'join_id_'+str(self.join_id)
                                    if waitkey_val == ord('.'):
                                        print ('using current join_id')
                                        self.track_metadata_dict[closest_track_index][flag_key] = 'join_id_'+str(self.join_id)
                            else:
                                self.track_metadata_dict[closest_track_index][flag_key] = flag_value

                if waitkey_val  == ord('r'):
                    print('remove flag from track, pos = {}'.format(self.mouse_event_data))
                    self.print_metadata_options()
                    flag_key, flag_value = self.handle_flag_options_waitkey()
                    if flag_key is not None:
                        try:
                            del self.track_metadata_dict[closest_track_index][flag_key]
                        except KeyError:
                            pass

                if waitkey_val == ord('s'):
                    print ('now saving the metadata dictionary of track flags')
                    metadata_json = json.dumps(self.track_metadata_dict)
                    self.metadata_fid.write('{0}\n'.format(metadata_json))

                self.mouse_event_flag = False
                self.mouse_event_data = {}

            else:
                # Normal key event - quit, move forward, backward, etc.
                if waitkey_val == ord('s'):
                    print ('now saving the metadata dictionary of track flags')
                    metadata_json = json.dumps(self.track_metadata_dict)
                    self.metadata_fid.write('{0}\n'.format(metadata_json))
                if waitkey_val == ord('q'):
                    frame_number = -1
                if waitkey_val == ord('f'):
                    frame_number += 1
                if waitkey_val == ord('b'):
                    frame_number -= 1
                ### kate addition 2018_05_21 #######
                if waitkey_val == ord('j'):
                    frame_number = int(raw_input("Jump to frame: "))
                    #frame_number = float(raw_input("Jump to frame: "))
                if waitkey_val == ord('n'): # <----- only works if you're not currently viewing a track
                    print ('jumping to next track')
                    frames_to_jump = 0
                    while len(self.frame_to_tracks_dict[frame_number +frames_to_jump]) == 0:
                        frames_to_jump +=1
                    frame_number = frame_number + frames_to_jump
                if waitkey_val == ord('o'): # <----- only works if you're not currently viewing a track
                    print ('jumping to next un-annotated track')
                    track_indices_to_jump = 0
                    step_back = 0
                    while len(self.frame_to_tracks_index_dict[frame_number-step_back]) ==0:
                        step_back+=1
                    prev_idx = int(self.frame_to_tracks_index_dict[frame_number-step_back][0]) #picking the first track index in the current frame
                    while len(self.track_metadata_dict[prev_idx+track_indices_to_jump]) !=0:
                        track_indices_to_jump +=1
                    index_to_jump_to = prev_idx +track_indices_to_jump
                    for frame_key in self.frame_to_tracks_index_dict:
                        if index_to_jump_to in self.frame_to_tracks_index_dict[frame_key]:
                            frame_number = frame_key

                if waitkey_val == ord('z'): # <----- only works if you're not currently viewing a track
                    print ('jumping to next track annotated as having been split, but without a join_id')
                    track_indices_to_jump = 0
                    step_back = 0
                    while len(self.frame_to_tracks_index_dict[frame_number-step_back]) ==0:
                        step_back+=1
                    prev_idx = int(self.frame_to_tracks_index_dict[frame_number-step_back][0]) #picking the first track index in the current frame

                    while len(self.track_metadata_dict[prev_idx+track_indices_to_jump]) != 0: #while the current track has some annotation
                        flag_dict = self.track_metadata_dict[prev_idx+track_indices_to_jump]

                        if 2 in flag_dict:
                            print ('2 in flag dict')
                            if flag_dict[2][0:7] == 'join_id':
                                track_indices_to_jump +=1
                                print ('already has a join id')
                            else:
                                print ('broken track without join id')
                                break
                        else:
                            track_indices_to_jump +=1

                    index_to_jump_to = prev_idx +track_indices_to_jump
                    for frame_key in self.frame_to_tracks_index_dict:
                        if index_to_jump_to in self.frame_to_tracks_index_dict[frame_key]:
                            frame_number = frame_key

                if waitkey_val == ord('p'): # <----- only works if you're not currently viewing a track
                    print ('jumping to previous track')
                    frames_to_step_back = 0
                    while len(self.frame_to_tracks_dict[frame_number -frames_to_step_back]) == 0:
                        frames_to_step_back +=1
                    frame_number = frame_number - frames_to_step_back

        else:
            waitkey_val = cv2.waitKey(1) & 0xFF
            if waitkey_val == ord('q'):
                break
            frame_number += 1

        return frame_number


    def handle_flag_options_waitkey(self):
        waitkey_val = cv2.waitKey(0) & 0xFF
        try:
            waitkey_int = int(chr(waitkey_val))
        except ValueError:
            waitkey_int = -1
        try:
            value = self.metadata_flag_dict[waitkey_int]
            key = waitkey_int
        except KeyError:
            value = None
            key = None
        return key, value

    # def handle_join_id_waitkey(self):
    #     waitkey_val = cv2.waitKey(0) & 0xFF
    #     try:
    #         waitkey

    def get_frame_to_tracks_dict(self, start_frame, end_frame):

        frame_to_tracks_dict = {}     # Frame to list of tracks (in that frame)
        frame_to_tracks_index_dict = {}  # Frame to list of track indices (in that frame)

        # Empty list until start_frame
        for i in range(start_frame):
            frame_to_tracks_dict[i] = []

        # Loop over frames and add all tracks which contains frame to list for
        # that frame.
        for i in range(start_frame, end_frame+1):
            frame_to_tracks_dict[i] = []
            frame_to_tracks_index_dict[i] = []
            for track_index, track in enumerate(self.track_list):
                if any([i==item['frame'] for item in track]):
                    frame_to_tracks_dict[i].append(track)
                    frame_to_tracks_index_dict[i].append(track_index)

        return frame_to_tracks_dict, frame_to_tracks_index_dict

    def draw_partial_line_seg(self,frame, blob0, blob1, radius, color):
        x0 = blob0['centroid_x']
        y0 = blob0['centroid_y']
        x1 = blob1['centroid_x']
        y1 = blob1['centroid_y']

        dx = x1 - x0
        dy = y1 - y0
        vec_len = numpy.sqrt(dx**2 + dy**2)
        ## kjl added this if/else to deal with the case in which apparently two points in successive frames are getting stitched into a track
        if vec_len == 0.0:
            print ('vec_len: ' + str(vec_len))
        else:
            ux = dx/vec_len
            uy = dy/vec_len

            x_circ = int(x0 + radius*ux)
            y_circ = int(y0 + radius*uy)

            x1 = int(x1)
            y1 = int(y1)

            cv2.line(frame,(x_circ, y_circ), (x1, y1), color)
            cv2.circle(frame, (x1, y1), self.param['point_radius'], color)

# Utility functions
#--------------------------------------------------------------------------------------------------

def get_closest_track_index(point, tracks_in_frame, tracks_index_in_frame):
    dist_and_index_list = []
    for i, track in zip(tracks_index_in_frame, tracks_in_frame):
        dist = point_to_track_distance(point,track)
        dist_and_index_list.append((dist,i))
    dist_and_index_list.sort()
    if len(dist_and_index_list) != 0:
        min_dist_index = dist_and_index_list[0][1]
        return min_dist_index


def filter_outlying_segments(track_list, multiplier=2, use_mad=False, use_std = False, angle_diff = numpy.pi/4, filter_floor_pix=50):

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
        if use_std:
            intrapair_step_sigma = numpy.std(step_array)
            intrapair_step_mean = numpy.mean(step_array)
        else:
            angle_array = numpy.unwrap(numpy.arctan2(diff_y, diff_x))
            #delta_angle_array = numpy.diff(angle_array)rad()
            #accel_array = numpy.diff(step_array)

        #for index, current_step_size in enumerate(step_array):
        for index in range(1,len(step_array)):
            current_step_size = step_array[index]
            if use_mad:
                if numpy.abs(current_step_size - intrapair_median) > max(intrapair_mad*multiplier, filter_floor_pix):
                    flagged_indices.append(index+1)
            if use_std:
                if numpy.abs(current_step_size - intrapair_step_mean) > max(intrapair_step_sigma*multiplier,filter_floor_pix):
                    flagged_indices.append(index+1)
                    print('i: {0}, avg: {1:1.2f}, std: {2:1.2f}, max: {3:1.2f}, {4}'.format(i,intrapair_step_mean, intrapair_step_sigma, step_array.max(), len(track)))
            else: #relative to last segment, is this segment uncharacteristically sized OR angled?
                if 1.0/multiplier > numpy.abs(float(current_step_size)/step_array[index-1]) or numpy.abs(float(current_step_size)/step_array[index-1]) > multiplier:
                    flagged_indices.append(index+1)
                elif numpy.abs(angle_array[index]-angle_array[index-1]) > angle_diff*numpy.pi/180:
                    flagged_indices.append(index+1)

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

def join_tracks(track_list,gap_multiplier = 0.5,max_tracks_to_join = 4):
    new_track_list = []
    m = max_tracks_to_join
    index = 0
    # if len(track_list) < m:
    #     new_track_list = track_list
    while len(track_list)> m:
        growing_track = track_list[index]
        for indices_ahead in [int(x) for x in numpy.linspace(1,m,m)]:
            tail_to_consider = track_list[index+indices_ahead]
            current_head = track_list[index+indices_ahead -1]
            frame_gap = tail_to_consider[0]['frame'] - current_head[-1]['frame']
            if  3 > frame_gap > 0: #if the adjacent tracks are 1 or 2 frames apart
                head_diffx  = current_head[-1]['blob']['centroid_x']-current_head[-2]['blob']['centroid_x']
                head_diffy  = current_head[-1]['blob']['centroid_y']-current_head[-2]['blob']['centroid_y']
                tail_diffx  = tail_to_consider[1]['blob']['centroid_x']-tail_to_consider[0]['blob']['centroid_x']
                tail_diffy  = tail_to_consider[1]['blob']['centroid_y']-tail_to_consider[0]['blob']['centroid_y']

                max_gap_to_bridge = gap_multiplier*(numpy.sqrt(head_diffx**2+head_diffy**2) + numpy.sqrt(tail_diffx**2 + tail_diffy**2))/2

                head_projection = [current_head[-1]['blob']['centroid_x']+head_diffx, current_head[-1]['blob']['centroid_y']+head_diffy ]
                tail_projection = [tail_to_consider[0]['blob']['centroid_x']-tail_diffx, tail_to_consider[0]['blob']['centroid_y']-tail_diffy]
                if frame_gap == 2: # this means there's a single intervening frame to bridge
                    if numpy.sqrt((head_projection[0]-tail_projection[0])**2 + (head_projection[1]-tail_projection[1])**2) < max_gap_to_bridge:
                        growing_track = growing_track+tail_to_consider
                        continue #keep considering growing the track
                    else:
                        break
                elif frame_gap == 1: # this means there's no frame to bridge
                    if numpy.sqrt((head_projection[0]-tail_to_consider[0]['blob']['centroid_x'])**2 + (head_projection[1]-tail_to_consider[0]['blob']['centroid_y'])**2) < max_gap_to_bridge:
                        growing_track = growing_track+tail_to_consider
                        continue #keep considering growing the track
                    else:
                        break
            else:
                break
        new_track_list.append(growing_track)
        del track_list[index:index+indices_ahead]
    new_track_list+= track_list[-m:] #adds the last few tracks that weren't subject to this splicing
    return new_track_list


# the version of join_tracks below is possibly not yet working; tries to take into account local curvature of tracks
# def join_tracks(track_list,gap_multiplier = 0.5,max_tracks_to_join = 4):
#     new_track_list = []
#     m = max_tracks_to_join
#     index = 0
#     while len(track_list)> m:
#         growing_track = track_list[index]
#         for indices_ahead in [int(x) for x in numpy.linspace(1,m,m)]:
#             tail_to_consider = track_list[index+indices_ahead]
#             current_head = track_list[index+indices_ahead -1]
#             frame_gap = tail_to_consider[0]['frame'] - current_head[-1]['frame']
#             if  3 > frame_gap > 0: #if the adjacent tracks are 1 or 2 frames apart
#                 head_diffx  = current_head[-1]['blob']['centroid_x']-current_head[-2]['blob']['centroid_x']
#                 head_diffy  = current_head[-1]['blob']['centroid_y']-current_head[-2]['blob']['centroid_y']
#                 tail_diffx  = tail_to_consider[1]['blob']['centroid_x']-tail_to_consider[0]['blob']['centroid_x']
#                 tail_diffy  = tail_to_consider[1]['blob']['centroid_y']-tail_to_consider[0]['blob']['centroid_y']
#                 #
#                 # head_last_ang = numpy.unwrap(numpy.arctan2(head_diffy, head_diffx))
#                 # tail_last_ang = numpy.unwrap(numpy.arctan2(head_diffy, head_diffx))
#                 head_positions = numpy.array([[x['blob']['centroid_x']for x in current_head],[y['blob']['centroid_y']for y in current_head]]).T
#                 tail_positions = numpy.array([[x['blob']['centroid_x']for x in tail_to_consider],[y['blob']['centroid_y']for y in tail_to_consider]]).T
#                 head_angles = numpy.unwrap([numpy.arctan2(a[1],a[0]) for a in head_positions])
#                 print (head_angles)
#                 tail_angles = numpy.unwrap([numpy.arctan2(a[1],a[0]) for a in tail_positions])
#                 head_ang_vel = numpy.diff(head_angles)
#                 tail_ang_vel = numpy.diff(tail_angles)
#
#                 max_gap_to_bridge = gap_multiplier*(numpy.sqrt(head_diffx**2+head_diffy**2) + numpy.sqrt(tail_diffx**2 + tail_diffy**2))/2
#
#                 head_projection_x = head_positions[-1][0] +    (numpy.cos(head_angles[-1] + head_ang_vel[-1])) * numpy.sqrt(head_diffx**2+head_diffy**2)
#                 head_projection_y = head_positions[-1][1] +    (numpy.sin(head_angles[-1] + head_ang_vel[-1])) * numpy.sqrt(head_diffx**2+head_diffy**2)
#                 head_projection = [head_projection_x, head_projection_y]
#
#                 tail_projection_x = tail_positions[0][0]  -    (numpy.cos(tail_angles [0] - tail_ang_vel[0]))  * numpy.sqrt(tail_diffx**2+tail_diffy**2)
#                 tail_projection_y = tail_positions[0][1]  -    (numpy.sin(tail_angles [0] - tail_ang_vel[0]))  * numpy.sqrt(tail_diffx**2+tail_diffy**2)
#                 tail_projection = [tail_projection_x, tail_projection_y]
#
#                 # head_projection = [current_head[-1]['blob']['centroid_x']+head_diffx, current_head[-1]['blob']['centroid_y']+head_diffy ]
#                 # tail_projection = [tail_to_consider[0]['blob']['centroid_x']-tail_diffx, tail_to_consider[0]['blob']['centroid_y']-tail_diffy]
#                 if frame_gap == 2: # this means there's a single intervening frame to bridge
#                     if numpy.sqrt((head_projection[0]-tail_projection[0])**2 + (head_projection[1]-tail_projection[1])**2) < max_gap_to_bridge:
#                         growing_track = growing_track+tail_to_consider
#                         continue #keep considering growing the track
#                     else:
#                         break
#                 elif frame_gap == 1: # this means there's no frame to bridge
#                     if numpy.sqrt((head_projection[0]-tail_to_consider[0]['blob']['centroid_x'])**2 + (head_projection[1]-tail_to_consider[0]['blob']['centroid_y'])**2) < max_gap_to_bridge:
#                         growing_track = growing_track+tail_to_consider
#                         continue #keep considering growing the track
#                     else:
#                         break
#             else:
#                 break
#         new_track_list.append(growing_track)
#         del track_list[index:index+indices_ahead]
#
#     return new_track_list



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

def point_to_blob_distance(point, blob):
    blob_x, blob_y = blob_position(blob)
    x,y = point
    return math.sqrt((blob_x - x)**2 + (blob_y - y)**2)

def point_to_track_distance(point, track):
    distance_list = [point_to_blob_distance(point,item['blob']) for item in track]
    return min(distance_list)
