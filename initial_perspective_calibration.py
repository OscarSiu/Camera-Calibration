import numpy as np
import cv2
import math
import yaml

writeValues=True

savedir="matrix/"

# Load previously saved data
with np.load('matrix\params.npz') as file:
    cam_mtx, dist= [file[i] for i in ('mtx','dist')]

with np.load('matrix\calibrated.npz') as file:
    roi, newcam_mtx= [file[i] for i in ('roi','newcam_mtx')]

#load center points from New Camera matrix
cx=newcam_mtx[0,2]
cy=newcam_mtx[1,2]
fx=newcam_mtx[0,0]
fy =newcam_mtx[1,1]
print("cx: "+str(cx)+",cy: "+str(cy)+",fx: "+str(fx)+",fy: "+str(fy))


#MANUALLY INPUT YOUR MEASURED POINTS HERE
#ENTER (X,Y,d*)
#d* is the distance from your point to the camera lens. (d* = Z for the camera center)
#we will calculate Z in the next steps after extracting the new_cam matrix

# at least 9 points
total_points_used=9 #world center + world points

#unit = centimeters
#X_center=16.1
#Y_center=14.52

X_center=98.736
Y_center= 33.275
Z_center= 300 # input from range finder
worldPoints=np.array([[X_center,Y_center,Z_center],
                        #[2.17,4.59,300],
                       #[19.11, 4.59, 300],
                      # [2.17, 26.37, 300],
                      # [19.11, 26.37, 300],
                      # [7.01, 11.85,300],
                      # [14.27, 11.85, 300],
                      # [7.01, 19.11, 300],
                      # [14.27, 19.11, 300]], dtype=np.float32)
                       
                       [2.178,4.84,300],
                       [19.118, 4.84, 300],
                       [2.178, 26.62, 300],
                       [19.118, 26.62, 300],
                       [7.018, 12.1,300],
                       [14.278, 12.1, 300],
                       [7.018, 19.36, 300],
                       [14.278, 19.36, 300]], dtype=np.float32)

#MANUALLY INPUT THE DETECTED IMAGE COORDINATES HERE

#[u,v] center + Image points
imagePoints=np.array([[cx,cy],
                       [982,1295],
                       [1154,1295],
                       [982,1523],
                       [1154,1524],
                       [1030, 1373],
                       [1105, 1373],
                       [1029, 1448],
                       [1106, 1448]], dtype=np.float32)

                       #[620,332],
                       #[674,332],
                       #[620,400],
                       #[674,400],
                       #[636, 355],
                       #[666, 355],
                       #[636, 385],
                       #[666, 385]], dtype=np.float32)


#FOR REAL WORLD POINTS, CALCULATE Z from d*

for i in range(1,total_points_used):
    #start from 1, given for center Z=d*
    #to center of camera
    wX=worldPoints[i,0]-X_center
    wY=worldPoints[i,1]-Y_center
    wd=worldPoints[i,2]

    d1=np.sqrt(np.square(wX)+np.square(wY))
    wZ=np.sqrt(np.square(wd)-np.square(d1))
    worldPoints[i,2]=wZ

#print('\n', worldPoints)

print("\nCamera matrix:\n", cam_mtx)
#print("\nDistortion coefficient:\n", dist)
#print("\nROI:\n", roi)
print("\nNew camera matrix: \n", newcam_mtx)

inverse_newcam_mtx = np.linalg.inv(newcam_mtx)
#print("\nInverse New Camera Matrix: \n", inverse_newcam_mtx)

if writeValues==True: np.save(savedir+'inverse_newcam_mtx.npy', inverse_newcam_mtx)

#print("\nCalibration Loaded")


#print("solvePNP")
ret, rvec1, tvec1=cv2.solvePnP(worldPoints, imagePoints,newcam_mtx,dist)

print("\nRotation vector:\n", rvec1)
if writeValues==True: np.save(savedir+'rvec1.npy', rvec1)

print("\nTranslation vector:\n", tvec1)
if writeValues==True: np.save(savedir+'tvec1.npy', tvec1)


R_mtx, jac=cv2.Rodrigues(rvec1)
#print("\nR - rodrigues vecs\n", R_mtx)
if writeValues==True: np.save(savedir+'R_mtx.npy', R_mtx)


Rt=np.column_stack((R_mtx,tvec1))
#print("\nExtrinsic Matrix\n", Rt)
if writeValues==True: np.save(savedir+'Rt.npy', Rt)


P_mtx=newcam_mtx.dot(Rt)
#print("\nnewCamMtx*R|t - Projection Matrix\n", P_mtx)
if writeValues==True: np.save(savedir+'P_mtx.npy', P_mtx)

########################### [XYZ1] to (u,v) ###########################################
s_arr=np.array([0], dtype=np.float32)
s_describe=np.array([0,0,0,0,0,0,0,0,0,0],dtype=np.float32)

GSD =[]

for i in range(0,total_points_used):
    print("\n=======POINT # " + str(i) +" =========================")
    
    print("Forward: From World Points, Find Image Pixel")
    XYZ1=np.array([[worldPoints[i,0],worldPoints[i,1],worldPoints[i,2],1]], dtype=np.float32)
    XYZ1=XYZ1.T

    print("\nXYZ1\n", XYZ1)
    suv1=P_mtx.dot(XYZ1)
    #print("\nsuv1\n", suv1)
    s=suv1[2,0]    
    uv1=suv1/s

    print("\nImage Points\n", uv1)
    print("\nScaling Factor\n", s)

    s_arr=np.array([s/total_points_used+s_arr[0]], dtype=np.float32)
    s_describe[i]=s
    if writeValues==True: np.save(savedir+'s_arr.npy', s_arr)

########################## Image pixels to (X,Y) coordinate #################################

    print("\nSolve: From Image Pixels, find World Points")

    uv_1=np.array([[imagePoints[i,0],imagePoints[i,1],1]], dtype=np.float32)
    uv_1=uv_1.T
    print("\nuv1\n", uv_1)
    suv_1=s*uv_1
    #print("\nsuv1\n", suv_1)

    #print("\nget camera coordinates, multiply by inverse Camera Matrix, subtract tvec1")
    xyz_c=inverse_newcam_mtx.dot(suv_1)
    xyz_c=xyz_c-tvec1
    #print("      xyz_c")
    inverse_R_mtx = np.linalg.inv(R_mtx)
    XYZ=inverse_R_mtx.dot(xyz_c)
    print("\nXYZw\n", XYZ)


    dist = math.dist([XYZ[0],XYZ[1]],[X_center,Y_center])
    pixel = math.dist([uv1[0],uv1[1]],[cx,cy])
    GSD.append(float(dist/pixel))

print('\nGSD: ', GSD)
avg_GSD = sum(GSD)/len(GSD)
print('\nAvg GSD: ', avg_GSD, 'cm/pixel')


#save it to a file
with open("output\gsd.yaml", "w") as f:
    yaml.dump(avg_GSD, f)

f.close()