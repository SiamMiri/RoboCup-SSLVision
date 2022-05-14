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

list_circle_position = {'1' : [41, 18], 
                        '2' : [19, 11], 
                        '3' : [12, 30], 
                        '4' : [31, 42]}

# Pint 2 and 3

point_one = 1



lst_min_line = {210: ["1","3"], 120:["1","2"],  point_one:["2","3"], 800:["4","3"]}



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
    print(angleD)


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
        if lst_min_line["1"][1] < lst_min_line["2"][1]:
            print("1 is up")

        else:
            print("2 is up")

    if lst_min_line_order[1] == "1" and lst_min_line_order[0] == "2":
        print("first assumption 2")



    ''' second assumption '''
    # # second assumption # #
    if lst_min_line_order[0] == "3" and lst_min_line_order[1] == "2":
        print("second assumption 1")
        # TODO: Check if added by 90 degree is correct in both side
        if lst_min_line["2"][0] < lst_min_line["3"][0]:
            if lst_min_line["2"][1] < lst_min_line["3"][1]:
                print("2 is up and BACK")
                print("3 is down and FRONT")
            else:
                print("This Condition is not possible !!")
        else:
            print("2 is up and FRONT")
            print("3 is down and BACK")
            getAngle(PT1= [lst_min_line["3"][0], lst_min_line["3"][1]],
                     PT2= [lst_min_line["2"][0], lst_min_line["2"][1]],
                     PT3= [lst_min_line["3"][0], lst_min_line["2"][1]])
            

    if lst_min_line_order[1] == "3" and lst_min_line_order[0] == "2":
        print("second assumption 2")
        if lst_min_line["2"][0] < lst_min_line["3"][0]:
            if lst_min_line["2"][1] < lst_min_line["3"][1]:
                print("2 is up and back")
                print("3 is down and frot")
                
                
            else:
                print("2 is down and back")
                print("3 is up and front")
                print("THIS CONFIG IS NOT POSSIBLE !!!")
                
        else:
            if lst_min_line["2"][1] < lst_min_line["3"][1]:
                print("2 is up and front")
                print("3 is down and back")
                # FIXED:
                getAngle(PT1= [lst_min_line["3"][0], lst_min_line["3"][1]],
                         PT2= [lst_min_line["2"][0], lst_min_line["2"][1]],
                         PT3= [lst_min_line["2"][0], lst_min_line["3"][1]])

    ''' third assumption '''
    # # third assumption # #
    if lst_min_line_order[0] == "3" and lst_min_line_order[1] == "4":
        print("third assumption")
        if lst_min_line["3"][1] < lst_min_line["4"][1]:
            if lst_min_line["3"][0] < lst_min_line["4"][0]:
                print("3 is up and back")
                print("4 is down and front")
                # TODO: Check the 10 degree error
            else:
                print("3 is down and back")
                print("4 is up and front")
                print("THIS IS NOT POSSIBLE!!")
        else:
            if lst_min_line["3"][0] > lst_min_line["4"][0]:
                print("3 is up and back")
                print("4 is down and front")
                print("THIS IS NOT POSSIBLE!!")

    if lst_min_line_order[1] == "3" and lst_min_line_order[0] == "4":
        print("third assumption")
        if lst_min_line["3"][1] < lst_min_line["4"][1]:
            if lst_min_line["3"][0] < lst_min_line["4"][0]:
                print("3 is up and back")
                print("4 is down and front")
                # TODO: Check the 10 degree error
            else:
                print("3 is down and back")
                print("4 is up and front")
                print("THIS IS NOT POSSIBLE!!")
        else:
            if lst_min_line["3"][0] > lst_min_line["4"][0]:
                print("3 is up and back")
                print("4 is down and front")
                print("THIS IS NOT POSSIBLE!!")
                
        ''' fourth assumption '''
        # # fourth assumption # #
    if lst_min_line_order[0] == "4"  and lst_min_line_order[1] == "2":
        print("fourth assumption")
        if lst_min_line_order[1] == "4" and lst_min_line_order[0] == "2":
            print("fourth assumption")




angle_between_circle(list(lst_min_line_order.items())[0][1], list_circle_position)


list_circle_position_1 = [[58, 29],
                          [17, 43],
                          [10, 23]]


list_circle_position_2 = [[11, 38],
                          [31, 42],
                          [31, 38]] # 11, 38
# 11, 38

list_circle_position_optimal = [[0, 0],
                                [35, 41],
                                [14, 41]]

# getAngle(list_circle_position_optimal)points = []
#img = np.zeros((512,512,3), np.uint8)

'''
img = cv2.imread('ee.jpg',3)
points = []
def draw(event, x,y, flags, params):

    if event==cv2.EVENT_LBUTTONDOWN:
        points.append([x, y])
        cv2.circle(img, (x,y), 5, (255,0,0), -1)
        if (len(points)!=0) :
            cv2.arrowedLine(img, tuple(points[0]),(x,y), (255,0,0),3)
        cv2.imshow('image', img)
        print(points)
        if len(points)%3==0:
            degrees = angle()

            print(abs(degrees))


def angle():
    a = points[-2]
    b = points[-3]
    c = points[-1]

    m1 = slope(b,a)
    m2 = slope(b,c)
    angle = math.atan((m2-m1)/1+m1*m2)
    angle = round(math.degrees(angle))
    if angle<0:
        angle = 180+angle
    cv2.putText(img, str((angle)), (b[0]-40,b[1]+40), cv2.FONT_HERSHEY_DUPLEX, 2,(0,0,255),2,cv2.LINE_AA)
    cv2.imshow('image',img)
    return angle

def slope(p1,p2):
    return (p2[1]-p1[1])/(p2[0]-p1[0])

while True:
    cv2.imshow('image', img)
    cv2.setMouseCallback('image', draw)
    if cv2.waitKey(1)&0xff==ord('r'):
        #img = np.zeros((512,512,3), np.uint8)
        img = cv2.imread('protractor.jpg', 3)
        points=[]
        cv2.imshow('image', img)
        cv2.setMouseCallback('image', draw)
    if cv2.waitKey(1)&0xff==ord('q'):
        break
'''