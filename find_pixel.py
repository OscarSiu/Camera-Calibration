import cv2
#from cv2 import INTER_AREA
import math

#define
img_path = 'Mini2/HR_cal/DJI_00873m.jpg'
GSD = 1.06 # (mm/pixel) dji mini2 4K
work_dis = 3 # (m)

#const
sample_dis = 3 # (m) sampling distance
font = cv2.FONT_HERSHEY_SIMPLEX

def click_event(event, x, y, flags, params):
	global img
	# checking for left mouse clicks
	if event == cv2.EVENT_LBUTTONDOWN:
		#cv2.putText(img, str(x) + ',' + str(y), (x,y), font, 2, color, 5)    # displaying the coordinates
		cv2.circle(img, (x,y), 2, (0,0,255), 5)
		print(x, ' ', y)
		cv2.imshow('image', img)
		global a
		a = (x,y)

	if event == cv2.EVENT_RBUTTONDOWN:
		#cv2.putText(img, str(x) + ',' + str(y), (x,y), font, 2, color, 5)  # displaying the coordinates
		cv2.circle(img, (x,y), 2, (0,255,0), 5)
		print(x, ' ', y)
		cv2.imshow('image', img)
		global b
		b = (x,y)

		cal_dis(a,b)

	# Reload image
	if event == cv2.EVENT_MBUTTONDOWN:
		cv2.namedWindow('image', cv2.WINDOW_NORMAL)		
		img = cv2.imread(img_path, 1)
		cv2.imshow('image', img)
		cv2.setMouseCallback('image', click_event)
		

#calculate real world distance and display
def cal_dis(pt1, pt2):
	len = math.dist(pt1,pt2)
	#print("pixel len: {:.2f}".format(len))
	#print("GSD: {:.4f}".format(GSD), "mm/pixel")

	obj_dist = '{:.2f}'.format((GSD * len * work_dis/sample_dis/10))
	print('dist: ', obj_dist, 'mm\n')
	midpt = (int((pt1[0]+pt2[0])/2),int((pt1[1]+pt2[1])/2))
	
	#display distance
	cv2.putText(img, str(obj_dist) + ' mm', midpt, font, 2, (0,0,255), 5)
	cv2.imshow('image', img)
	

#Main
if __name__=="__main__":

	cv2.namedWindow('image', cv2.WINDOW_NORMAL)
	img = cv2.imread(img_path, 1)
	cv2.imshow('image', img)
	
	# setting mouse handler for the image and trigger click_event() function
	cv2.setMouseCallback('image', click_event)

	cv2.waitKey(0)
	cv2.destroyAllWindows()