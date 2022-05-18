# These Codes are non Git Codes
# 
# @Sia 12.Mai.2022

import collections
import cv2
import math



'''

num_x_cor green: [10]

num_y_cor green: [30]

num_x_cor red: [32, 45, 19]

num_y_cor red: [46, 17, 11]

'''



frame = [58,58, 3]



#circle_pos: [x, y]
list_circle_position = [[19, 11],
                        [12, 30],
                        [12, 11]] 

list_circle_position = {'1' : [31,  7], 
                        '2' : [11, 12], 
                        '3' : [11, 38], 
                        '4' : [41, 33]}

# Pint 2 and 3

point_one = 1



lst_min_line = {985: ["1","3"], point_one:["1","2"],  800:["2","3"], 500:["4","3"], 1235: ["4","1"]}



lst_min_line_order = collections.OrderedDict(sorted(lst_min_line.items()))



# print(next(iter(lst_min_line_order)))



''' Now we have the list in the order and with the Position of the Circle Color'''

''' We have to pass the whole package with this list to find with of the circle is

in the lowst part for calculation of the angle '''

def gradient(pt1 ,pt2):
    return(pt2[1] - pt1[1])/ (pt2[0] - pt1[0]) 

def getAngle(PT1 = None, PT2 = None, PT3 = None):
    pt1 = PT1
    pt2 = PT2
    pt3 = PT3
    print(pt1, pt2, pt3)
    n1 = gradient(pt1 , pt2)
    n2 = gradient(pt1 ,pt3)
    angleR = math.atan((n2 - n1)/(1 + (n2*n1)))
    angleD = round(math.degrees(angleR))
    return angleD


def angle_between_circle(lst_min_line_order = None, lst_min_line =  None):

    print(lst_min_line_order)
    print(lst_min_line)

    ''' Durchmesser ist Ungültig '''

    if lst_min_line_order[0] == '1' and lst_min_line_order[0] == '3':
        return None

    if lst_min_line_order[0] == '3' and lst_min_line_order[0] == '1':

        return None

    if lst_min_line_order[0] == '2' and lst_min_line_order[0] == '4':

        return None

    if lst_min_line_order[0] ==  '4' and lst_min_line_order[0] == '2':

        return None


    ''' first assumption '''
    # # first assumption # #
    if lst_min_line_order[0] == "1" and lst_min_line_order[1] == "2":
        print("first assumption 1")
        if lst_min_line["1"][0] > lst_min_line["2"][0]:
            if lst_min_line["1"][1] > lst_min_line["2"][1]:
                print("1 is right and down")
                print("2 is left and up")
                Angle = getAngle(PT1 = [lst_min_line["2"][0], lst_min_line["2"][1]], # 
                                 PT2 = [lst_min_line["1"][0], lst_min_line["1"][1]], #
                                 PT3 = [lst_min_line["1"][0], lst_min_line["2"][1]])
                Angle = Angle + 180 
                print(Angle)
                return Angle
            else:
                print("1 is right and up")
                print("2 is left and down")
                Angle = getAngle(PT1 = [lst_min_line["2"][0], lst_min_line["2"][1]], # 
                                 PT2 = [lst_min_line["1"][0], lst_min_line["1"][1]], #
                                 PT3 = [lst_min_line["1"][0], lst_min_line["2"][1]])
                Angle = 180 - Angle 
                print(Angle)
                return Angle

        else:
            print("THIS CONFIG IS NOT POSSIBLE !!!")

    if lst_min_line_order[1] == "1" and lst_min_line_order[0] == "2":
        print("first assumption 2")
        if lst_min_line["1"][0] > lst_min_line["2"][0]:
            if lst_min_line["1"][1] > lst_min_line["2"][1]:
                print("1 is right and down")
                print("2 is left and up")
                Angle = getAngle(PT1 = [lst_min_line["2"][0], lst_min_line["2"][1]], # 
                                 PT2 = [lst_min_line["1"][0], lst_min_line["1"][1]], #
                                 PT3 = [lst_min_line["1"][0], lst_min_line["2"][1]])
                Angle = Angle + 180 
                print(Angle)
                return Angle
            else:
                print("1 is right and up")
                print("2 is left and down")
                Angle = getAngle(PT1 = [lst_min_line["2"][0], lst_min_line["2"][1]], # 
                                 PT2 = [lst_min_line["1"][0], lst_min_line["1"][1]], #
                                 PT3 = [lst_min_line["1"][0], lst_min_line["2"][1]])
                Angle = 180 - Angle 
                print(Angle)
                return Angle

        else:
            print("THIS CONFIG IS NOT POSSIBLE !!!")


    ''' second assumption '''
    # # second assumption # #
    if lst_min_line_order[0] == "3" and lst_min_line_order[1] == "2":
        print("second assumption 1")
        if lst_min_line["2"][0] < lst_min_line["3"][0]:
            if lst_min_line["2"][1] < lst_min_line["3"][1]:
                print("2 is up and back")
                print("3 is down and frot")
                # FIXED
                Angle = getAngle(PT1 = [lst_min_line["3"][0], lst_min_line["3"][1]], # 
                                 PT2 = [lst_min_line["2"][0], lst_min_line["2"][1]], #
                                 PT3 = [lst_min_line["2"][0], lst_min_line["3"][1]])
                Angle = Angle * -1
                print(Angle)
                return Angle
                
                
            else:
                print("2 is down and back")
                print("3 is up and front")
                print("THIS CONFIG IS NOT POSSIBLE !!!")
                
        else:
            if lst_min_line["2"][1] < lst_min_line["3"][1]:
                print("2 is up and front")
                print("3 is down and back")
                # FIXED
                Angle = getAngle(PT1 = [lst_min_line["3"][0], lst_min_line["3"][1]], # 
                                 PT2 = [lst_min_line["2"][0], lst_min_line["2"][1]], #
                                 PT3 = [lst_min_line["2"][0], lst_min_line["3"][1]])
                Angle = 90 - Angle + 90
                print(Angle)
                return Angle
            

    if lst_min_line_order[1] == "3" and lst_min_line_order[0] == "2":
        print("second assumption 2")
        if lst_min_line["2"][0] < lst_min_line["3"][0]:
            if lst_min_line["2"][1] < lst_min_line["3"][1]:
                print("2 is up and back")
                print("3 is down and frot")
                # FIXED
                Angle = getAngle(PT1 = [lst_min_line["3"][0], lst_min_line["3"][1]], # 
                                 PT2 = [lst_min_line["2"][0], lst_min_line["2"][1]], #
                                 PT3 = [lst_min_line["2"][0], lst_min_line["3"][1]])
                Angle = Angle * -1
                print(Angle)
                return Angle
                
                
            else:
                print("2 is down and back")
                print("3 is up and front")
                print("THIS CONFIG IS NOT POSSIBLE !!!")
                
        else:
            if lst_min_line["2"][1] < lst_min_line["3"][1]:
                print("2 is up and front")
                print("3 is down and back")
                # FIXED
                Angle = getAngle(PT1 = [lst_min_line["3"][0], lst_min_line["3"][1]], # 
                                 PT2 = [lst_min_line["2"][0], lst_min_line["2"][1]], #
                                 PT3 = [lst_min_line["2"][0], lst_min_line["3"][1]])
                Angle = 90 - Angle + 90
                print(Angle)
                return Angle
            
    ''' third assumption '''
    # # third assumption # #
    if lst_min_line_order[0] == "3" and lst_min_line_order[1] == "4":
        print("third assumption 1")
        if lst_min_line["3"][0] < lst_min_line["4"][0]:
            if lst_min_line["3"][1] < lst_min_line["4"][1]:
                print("3 is up and back")
                print("4 is down and front")
                # TODO: Check the 10 degree error
                # FIXED
                Angle = getAngle(PT1 = [lst_min_line["4"][0], lst_min_line["4"][1]], # 
                                 PT2 = [lst_min_line["3"][0], lst_min_line["3"][1]], #
                                 PT3 = [lst_min_line["3"][0], lst_min_line["4"][1]])
                Angle = Angle * -1
                print(Angle)
                return Angle
            else:
                print("3 is down and back")
                print("4 is up and front")
                # FIXED
                Angle = getAngle(PT1 = [lst_min_line["4"][0], lst_min_line["4"][1]], # 
                                 PT2 = [lst_min_line["3"][0], lst_min_line["3"][1]], #
                                 PT3 = [lst_min_line["3"][0], lst_min_line["4"][1]])
                Angle = Angle * -1
                print(Angle)
                return Angle
        else:
            print("THIS IS NOT POSSIBLE!!")

    if lst_min_line_order[1] == "3" and lst_min_line_order[0] == "4":
        print("third assumption 2")
        if lst_min_line["3"][0] < lst_min_line["4"][0]:
            if lst_min_line["3"][1] < lst_min_line["4"][1]:
                print("3 is up and back")
                print("4 is down and front")
                # TODO: Check the 10 degree error
                # FIXED
                Angle = getAngle(PT1 = [lst_min_line["4"][0], lst_min_line["4"][1]], # 
                                 PT2 = [lst_min_line["3"][0], lst_min_line["3"][1]], #
                                 PT3 = [lst_min_line["3"][0], lst_min_line["4"][1]])
                Angle = Angle * -1
                print(Angle)
                return Angle
            else:
                print("3 is down and back")
                print("4 is up and front")
                # FIXED
                Angle = getAngle(PT1 = [lst_min_line["4"][0], lst_min_line["4"][1]], # 
                                 PT2 = [lst_min_line["3"][0], lst_min_line["3"][1]], #
                                 PT3 = [lst_min_line["3"][0], lst_min_line["4"][1]])
                Angle = Angle * -1
                print(Angle)
                return Angle
        else:
            print("THIS IS NOT POSSIBLE!!")
                
        ''' fourth assumption '''
        # # fourth assumption # #
    if lst_min_line_order[0] == "4"  and lst_min_line_order[1] == "1":
        print("fourth assumption 1")
        if lst_min_line["1"][0] > lst_min_line["4"][0]:
            if lst_min_line["1"][1] < lst_min_line["4"][1]:
                print("1 is up and front")
                print("4 is down and back")
                # FIXED
                Angle = getAngle(PT1 = [lst_min_line["4"][0], lst_min_line["4"][1]], # 
                                 PT2 = [lst_min_line["1"][0], lst_min_line["1"][1]], #
                                 PT3 = [lst_min_line["1"][0], lst_min_line["4"][1]])
                Angle = Angle * -1
                print(Angle)
                return Angle
            else:
                print("THIS IS NOT POSSIBLE!!")
        else:
            print("1 is up and back")
            print("4 is down and front")
            print("1 is up and front")
            print("4 is down and back")
            # FIXED
            Angle = getAngle(PT1 = [lst_min_line["4"][0], lst_min_line["4"][1]], # 
                                PT2 = [lst_min_line["1"][0], lst_min_line["1"][1]], #
                                PT3 = [lst_min_line["1"][0], lst_min_line["4"][1]])
            Angle = ( Angle + 90 + 90 ) * -1
            print(Angle)
            return Angle
    
    if lst_min_line_order[1] == "4" and lst_min_line_order[0] == "1":
        print("fourth assumption 2")
        if lst_min_line["1"][0] > lst_min_line["4"][0]:
            if lst_min_line["1"][1] < lst_min_line["4"][1]:
                print("1 is up and front")
                print("4 is down and back")
                # FIXED
                Angle = getAngle(PT1 = [lst_min_line["4"][0], lst_min_line["4"][1]], # 
                                 PT2 = [lst_min_line["1"][0], lst_min_line["1"][1]], #
                                 PT3 = [lst_min_line["1"][0], lst_min_line["4"][1]])
                Angle = Angle * -1
                print(Angle)
                return Angle
            else:
                print("THIS IS NOT POSSIBLE!!")
        else:
            print("1 is up and back")
            print("4 is down and front")
            print("1 is up and front")
            print("4 is down and back")
            # FIXED
            Angle = getAngle(PT1 = [lst_min_line["4"][0], lst_min_line["4"][1]], # 
                                PT2 = [lst_min_line["1"][0], lst_min_line["1"][1]], #
                                PT3 = [lst_min_line["1"][0], lst_min_line["4"][1]])
            Angle = ( Angle + 90 + 90 ) * -1
            print(Angle)
            return Angle




angle_between_circle(list(lst_min_line_order.items())[0][1], list_circle_position)