from __future__ import print_function
import cv2
import numpy as np

class BlobFinder:

    def __init__(self, filter_by_area=True, min_area=100, max_area=None, open_kernel_size=3, close_kernel_size=3,kernel_shape='ellipse'):
        self.filter_by_area = filter_by_area 
        self.min_area = min_area 
        self.max_area = max_area 
        self.open_kernel_size = open_kernel_size 
        self.close_kernel_size = close_kernel_size
        self.circ_min_radius = 10
        self.circ_radius_margin = 15
        self.kernel_shape = kernel_shape

    def find(self, image, fg_mask):

        dummy, contour_list, dummy = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        fg_mask_copy = np.array(fg_mask)

        if self.open_kernel_size[0]*self.open_kernel_size[1] > 0: # <------- open_kernel_size isn't a list, I thought? 

            
            if self.kernel_shape == 'rect':
                open_kernel = np.ones(self.open_kernel_size, np.uint8)
                close_kernel = np.ones(self.close_kernel_size, np.uint8)
            else:
                open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,self.open_kernel_size) 
                close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,self.close_kernel_size)
                open_kernel = np.logical_or(open_kernel,open_kernel.T).astype(np.uint8)
                close_kernel = np.logical_or(close_kernel,close_kernel.T).astype(np.uint8)

            #------ Will is using this opening to remove higher-intensity noise
            fg_mask_copy = cv2.morphologyEx(fg_mask_copy,cv2.MORPH_OPEN,open_kernel)
            #---- KJL 2017_12_04 noticed some flies are split into two; maybe a "closing" operation will be helpful here            
            fg_mask_copy = cv2.morphologyEx(fg_mask_copy,cv2.MORPH_CLOSE,close_kernel)
            #-----------------------------------------------------------------------------------------------------------
        # Find blob data
        blob_list = []
        blob_contours = []

        for contour in contour_list:

            blob_ok = True

            # Get area and apply area filter  
            area = cv2.contourArea(contour)
            if self.filter_by_area:
                if area <= 0:
                    blob_ok = False
                if self.min_area is not None:
                    if area < self.min_area:
                        blob_ok = False
                if self.max_area is not None:
                    if area > self.max_area:
                        blob_ok = False

            # Get centroid
            moments = cv2.moments(contour)
            if moments['m00'] > 0 and blob_ok:
                centroid_x = moments['m10']/moments['m00']
                centroid_y = moments['m01']/moments['m00']
            else:
                blob_ok = False
                centroid_x = 0.0
                centroid_y = 0.0

            # Get bounding rectangle
            if blob_ok:
                bound_rect = cv2.boundingRect(contour)
                min_x = bound_rect[0]
                min_y = bound_rect[1]
                max_x = bound_rect[0] + bound_rect[2] 
                max_y = bound_rect[1] + bound_rect[3] 
            else:
                min_x = 0.0 
                min_y = 0.0
                max_x = 0.0
                max_y = 0.0
        
            #----------------KJL---------------------
            # check if blob is very close to another blob with respect to blob length; 
            # Create blob dictionary
            blob = {
                    'centroid_x' : centroid_x,
                    'centroid_y' : centroid_y,
                    'min_x'      : min_x,
                    'max_x'      : max_x,
                    'min_y'      : min_y,
                    'max_y'      : max_y,
                    'area'       : area,
                    } 
            
            # Ifblob is OK add to list of blobs
            if blob_ok: 
                blob_list.append(blob)
                blob_contours.append(contour)

        # Draw blob on image
        blob_image = cv2.cvtColor(fg_mask_copy,cv2.COLOR_GRAY2BGR)
        cv2.drawContours(blob_image,blob_contours,-1,(0,0,255),3)

        # Draw tracking circles on image
        circ_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) 
        for blob in blob_list:
            dx = abs(blob['max_x'] - blob['min_x'])
            dy = abs(blob['max_y'] - blob['min_y'])
            max_dim = max(dx,dy)
            circ_radius = max_dim + self.circ_radius_margin
            circ_radius =int( max(self.circ_min_radius, circ_radius))
            circ_pos = (int(blob['centroid_x']), int(blob['centroid_y']))
            circ_color = (0,0,np.iinfo(circ_image.dtype).max)
            circ_image = cv2.circle(circ_image, circ_pos, circ_radius, circ_color)

        return blob_list, blob_image, circ_image



