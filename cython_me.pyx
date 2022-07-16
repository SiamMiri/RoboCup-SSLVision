
import cv2

def loop_blue_circle(int lenContoursBlue, contours_blue, float area_of_circle_max, float area_of_circle_min, find_red_green_circle):
    cdef int x               = 0
    cdef float blue_area     = 0
    # cdef list moment         = 0
    cdef int cx_blue         = 0
    cdef int cy_blue         = 0
    cdef int blue_color_num  = 0
    for x in range(0,lenContoursBlue):
                
        blue_area = cv2.contourArea(contours_blue[x])
        if blue_area == 0.0:
            continue

        if blue_area < area_of_circle_max and blue_area > area_of_circle_min:                
            
            moment = cv2.moments(contours_blue[x]) 

            cx_blue = int(moment["m10"]/moment["m00"])
            cy_blue = int(moment["m01"]/moment["m00"])

            # crop_img = self._crop_robot_circle(frame, cy_blue, cx_blue, if_is_ball)
            # crop_img = self.creat_circle_color_id_mask(crop_img, color = "blue", cordinate_list=[cx_blue, cy_blue])
            
            # self._find_red_green_circle(crop_img, cy_blue, cx_blue, blue_color_dict) # , blue_color_num
            # pool.apply_async(self._find_red_green_circle, args=[crop_img, cy_blue, cx_blue, blue_color_dict])
            blue_color_num          += 1
            blue_color_dict         = f'Blue_Circle_{blue_color_num}'
            find_red_green_circle(cy_blue, cx_blue, blue_color_dict)