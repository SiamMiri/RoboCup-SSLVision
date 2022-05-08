import cv2
from matplotlib import pyplot as plt
import functools
import math
import numpy as np

bwimshow = functools.partial(plt.imshow, vmin=0, vmax=255,
                             cmap=plt.get_cmap('gray'))

def rotate_about_center(src, angle, widthOffset=0., heightOffset=0, scale=1.):
    w = src.shape[1]
    h = src.shape[0]

    # Add offset to correct for center of images.
    wOffset = -0.5 
    hOffset = -0.5

    rangle = np.deg2rad(angle)  # angle in radians
    # now calculate new image width and height
    nw = (abs(np.sin(rangle)*h) + abs(np.cos(rangle)*w))*scale
    nh = (abs(np.cos(rangle)*h) + abs(np.sin(rangle)*w))*scale
    print("nw = ", nw, "nh = ", nh)
    # ask OpenCV for the rotation matrix
    rot_mat = cv2.getRotationMatrix2D((nw*0.5 + wOffset, nh*0.5 + hOffset), angle, scale)
    # calculate the move from the old center to the new center combined
    # with the rotation
    rot_move = np.dot(rot_mat, np.array([(nw-w)*0.5 + widthOffset, (nh-h)*0.5 + heightOffset,0]))
    # the move only affects the translation, so update the translation
    # part of the transform
    rot_mat[0,2] += rot_move[0]
    rot_mat[1,2] += rot_move[1]
    return cv2.warpAffine(src, rot_mat, (int(math.ceil(nw)), int(math.ceil(nh))), flags=cv2.INTER_LANCZOS4)

def main():
    # create image
    rows = 10
    cols = 10
    angle = -90  
    widthOffset = 0  # need 1 to match 90 degrees and ? for -90 degrees.
    heightOffset = 0
    img = np.zeros((rows,cols), np.float32)
    img = cv2.imread("FieldTest_Left_Light_On_Daylight(hight).jpg")
    img[:, 0] = 255
    img[:, cols-1] = 255

    img[0, :] = 200
    img[rows-1, :] = 200

    # mark some pixels for reference points.
    img[0, int(cols/2 - 1)] = 0  
    img[rows-1, int(cols/2) - 1] = 100

    bwimshow(img)
    plt.show()

    img = rotate_about_center(img, angle, widthOffset, heightOffset)
    print("img shape = ", img.shape)
    print('Data type', img.dtype)
    bwimshow(img)
    plt.show()
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()