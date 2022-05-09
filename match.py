import cv2
import numpy as np
from matplotlib import pyplot as plt

dire = cv2.imread('tt.png')
dire_template = cv2.imread('dowdnload.png')
radiant = cv2.imread('tt.png')
radiant_template = cv2.imread('download.png')

# color images are in the form BGR
# removing the B and G from the images makes the "continue" button more distinct between the two teams
# since dire is red while radiant is green
dire_red = dire.copy()
dire_red[:,:,0] = 0
dire_red[:,:,1] = 0

dire_template_red = dire_template.copy()
dire_template_red[:,:,0] = 0
dire_template_red[:,:,1] = 0

radiant_red = radiant.copy()
radiant_red[:,:,0] = 0
radiant_red[:,:,1] = 0

radiant_template_red = radiant_template.copy()
radiant_template_red[:,:,0] = 0
radiant_template_red[:,:,1] = 0

dire_gray = cv2.cvtColor(dire_red, cv2.COLOR_BGR2GRAY)
dire_template_gray = cv2.cvtColor(dire_template_red, cv2.COLOR_BGR2GRAY)
radiant_gray = cv2.cvtColor(radiant_red, cv2.COLOR_BGR2GRAY)
radiant_template_gray = cv2.cvtColor(radiant_template_red, cv2.COLOR_BGR2GRAY)

# plt.figure(0)
# plt.imshow(dire_red)
# plt.figure(1)
# plt.imshow(radiant_red)
# plt.figure(2)
# plt.imshow(dire_gray, cmap='gray')
# plt.figure(3)
# plt.imshow(radiant_gray, cmap='gray')
# plt.figure(4)
# plt.imshow(dire_template_red)
# plt.figure(5)
# plt.imshow(radiant_template_red)
# plt.figure(6)
# plt.imshow(dire_template_gray)
# plt.figure(7)
# plt.imshow(radiant_template_gray, cmap='gray')

# plt.show()

w, h = dire_template_gray.shape[::-1]

# All the 6 methods for comparison in a list
methods = ['cv2.TM_CCOEFF_NORMED', 
            'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF_NORMED']

for meth in methods:
    print(f'{meth}: ')
    # this would be the live image
    img = dire_gray.copy()
    method = eval(meth)

    # Apply template Matching
    dire_res = cv2.matchTemplate(img,dire_template_gray,method)
    radiant_res = cv2.matchTemplate(img,radiant_template_gray,method)


    dire_vals = [min_val, max_val, min_loc, max_loc] = cv2.minMaxLoc(dire_res)
    radiant_vals = [min_val, max_val, min_loc, max_loc] = cv2.minMaxLoc(radiant_res)

    print(dire_vals)
    print(radiant_vals)
    # print(f'min val: {min_val} max val: {max_val}')

    # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
    if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
        top_left = min_loc
    else:
        top_left = max_loc
    bottom_right = (top_left[0] + w, top_left[1] + h)

    cv2.rectangle(img,top_left, bottom_right, 255, 2)

    # plt.subplot(121),plt.imshow(res,cmap = 'gray')
    plt.subplot(121),plt.imshow(dire_res)
    plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
    # plt.subplot(122),plt.imshow(img,cmap = 'gray')
    plt.subplot(122),plt.imshow(img)
    plt.title('Detected Point'), plt.xticks([]), plt.yticks([])
    plt.suptitle(meth)

    plt.show()