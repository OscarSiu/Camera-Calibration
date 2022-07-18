from cv2 import INTER_AREA
import numpy as np
import cv2 as cv
import glob
import yaml


################ FIND CHESSBOARD CORNERS - OBJECT POINTS AND IMAGE POINTS #############################

chessboardSize = (6,8)
h,  w = (2880,3840)

# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((chessboardSize[0] * chessboardSize[1], 3), np.float32)

#Account for 2.42 cm per square in grid
objp[:,:2] = np.mgrid[0:chessboardSize[0],0:chessboardSize[1]].T.reshape(-1,2)*2.42

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

#  Starts camera and check each frame for required pattern
images = glob.glob('M2EA\*.JPG')
found =0

for fname in images:
    print(fname)
    img = cv.imread(fname)
    #img = cv.resize(img,(1920,1080),interpolation=INTER_AREA)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv.findChessboardCorners(gray, chessboardSize, None)

    # If found, add object points, image points (after refining them)
    if ret == True:

        objpoints.append(objp)
        corners2 = cv.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
        imgpoints.append(corners2)

        # Draw and display the corners
        img = cv.drawChessboardCorners(img, chessboardSize, corners2, ret)
        found +=1
       # cv.imshow('verify', img)
       # cv.imwrite('output\det_corners' + str(found) +'.png', img)
       # cv.waitKey(500)

print("Number of images used for calibration: ", found)

cv.destroyAllWindows()

############## CALIBRATION #######################################################
ret, cameraMatrix, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)

#print('Camera calibrated: ', ret)
print("\nCamera matrix:\n", cameraMatrix)
#print("\nDistortion coefficient:\n", dist)
#print("\nRotation Vectors:\n", rvecs)
#print("\nTranslation Vectors:\n", tvecs)

np.savez("matrix\params.npz", mtx = cameraMatrix, dist=dist, rvecs = rvecs, tvecs = tvecs)

# transform the matrix and distortion coefficients to writable lists
data = {'camera_matrix': np.asarray(cameraMatrix).tolist(),
        'dist_coeff': np.asarray(dist).tolist(),
        'Rotational vectors': np.asarray(rvecs).tolist(),
        'Translational vectors': np.asarray(tvecs).tolist()}

#save it to a file
with open("matrix\calibration_matrix.yaml", "w") as f:
    yaml.dump(data, f)

f.close()



############## UNDISTORTION #####################################################

img = cv.imread('M2EA\DJI_0176_W.jpg')

#if use alpha 0, we discard the black pixels from distortion. 
# This helps to make the entire ROI = full dimensions of the image (after undistort)
#if use alpha 1, we retain the black pixels, and obtain the ROI as valid pixels for the matrix.
newCameraMatrix, roi = cv.getOptimalNewCameraMatrix(cameraMatrix, dist, (w,h), 1, (w,h))
#print('\nROI: ', roi)
print('\nNew Camera Matrix: \n', newCameraMatrix)
np.savez("matrix\calibrated.npz", roi = roi, newcam_mtx=newCameraMatrix)

#inverse = np.linalg.inv(newCameraMatrix)
#print("\nInverse New Camera Matrix: \n", inverse)

# Reprojection Error 
mean_error = 0

for i in range(len(objpoints)):
    imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], cameraMatrix, dist)
    error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2)/len(imgpoints2)
    mean_error += error

print("\ntotal error: {}".format(mean_error/len(objpoints)))

# Undistort (shortest path)
undst = cv.undistort(img, cameraMatrix, dist, None, newCameraMatrix)

# crop the image
x, y, w, h = roi
undst = undst[y:y+h, x:x+w]
cv.imwrite('output\caliResult1.png', undst)

# Undistort with Remapping (curved path)
mapx, mapy = cv.initUndistortRectifyMap(cameraMatrix, dist, None, newCameraMatrix, (w,h), 5)
dst = cv.remap(img, mapx, mapy, cv.INTER_LINEAR)

# crop the image
x, y, w, h = roi
dst = dst[y:y+h, x:x+w]
cv.imwrite('output\caliResult2.png', dst)

print("\nCalibration completed.\n")