import cv2
import numpy as np
import os

path = "Robo_ID"
orb = cv2.ORB_create(nfeatures=1000)


images = []
classNames = []
ID_List = os.listdir(path)
print(ID_List)
print("Total Robots Detected: ", len(ID_List))

nn = 1
for Idx in ID_List:
    # Idx.replace('.png', '')
    # Idx = Idx[4:]
    imgCur = cv2.imread(f'{path}/Robo{nn}.png')
<<<<<<< HEAD
    images.append(imgCur)
    #classNames.append(os.path.splitext(Idx)[0])
    classNames.append(f'Robo{nn}')
=======
    if imgCur is not None:
        #imgCur = cv2.resize(imgCur, (58,58))
        cv2.imshow(f"frame{nn}", imgCur)
        #print(len(imgCur))
    images.append(imgCur)
    classNames.append(os.path.splitext(Idx)[0])
    #classNames.append(f'Robo{nn}')
>>>>>>> bd61e1a7f4ee447e26f0532ed44c58a99db19f81
    nn = nn + 1
print(classNames)

def findDes(imges):
    desList = []
    for img in images:
        kp, des = orb.detectAndCompute(img, None)
        desList.append(des)
    return desList

def find_id(img, desList):
    kp2, des2 = orb.detectAndCompute(img, None)
    bf = cv2.BFMatcher()
    matchList = []
    finalVal = -1
    try:
<<<<<<< HEAD

=======
>>>>>>> bd61e1a7f4ee447e26f0532ed44c58a99db19f81
        for des in desList:
            matches = bf.knnMatch(des, des2, k=2)

            fine = []
            for m,n in matches:
                #if m.distance < 0.4 * n.distance: #0.75
                    #fine.append([m])
                fine.append([m])
            matchList.append(len(fine))

    except Exception as e:
        print(e)

    print(matchList)
    if len(matchList) is not 0:
        if max(matchList) > 0:
            finalVal = matchList.index(max(matchList))
    
    return finalVal

desList = findDes(images)
while True:
    #frame2 = cv2.imread("FieldTest_Left_Light_On_Daylight(hight).jpg")
<<<<<<< HEAD
    frame2 = cv2.imread("tttt.png")

    imgOrg = frame2.copy()
    frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
=======
    frame2 = cv2.imread("Robo1.png")
    print(len(frame2))
        # Create HOG Descriptor object
    hog = cv2.HOGDescriptor()

    # im = cv2.imread("Robot5_x1101_y299.jpg", 0) # Grayscale image

    # Compute HOG descriptor
    #h = hog.compute(im)

    imgOrg = frame2.copy()
    # frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
>>>>>>> bd61e1a7f4ee447e26f0532ed44c58a99db19f81

    id = find_id(imgOrg, desList)

    if id is not -1 :
        cv2.putText(imgOrg, classNames[id],(50,50),cv2.FONT_HERSHEY_SCRIPT_COMPLEX, 1,(0,0,255),1)

<<<<<<< HEAD
    imgOrg = cv2.resize(imgOrg, (740,420))
    cv2.imshow("frame", imgOrg)
    cv2.waitKey(1)
=======
    # imgOrg = cv2.resize(imgOrg, (740,420))
    cv2.imshow("frame", imgOrg)
    k = cv2.waitKey(1)
    if k % 256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break
>>>>>>> bd61e1a7f4ee447e26f0532ed44c58a99db19f81

# img1 = cv2.imread("temp.jpg")

# img2 = cv2.imread("Robot5_x323_y158.jpg")


# orb = cv2.ORB_create(nfeatures=1000)


# kp1, des1 = orb.detectAndCompute(img1, None)
# kp2, des2 = orb.detectAndCompute(img2, None)

# img3 = cv2.drawMatchesKnn(img1, kp1, img2,kp2, fine, None, flags=2)

# cv2.imshow("img1", img1)
# cv2.imshow("img2", img2)
# cv2.imshow("img3", img3)

# cv2.waitKey(0)