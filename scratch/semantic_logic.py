import cv2
import numpy as np
import itertools
import pandas as pd
from scipy import stats

def points_between(p1, p2):
    xs = range(p1[0] + 1, p2[0]) or [p1[0]]
    ys = range(p1[1] + 1, p2[1]) or [p1[1]]
    return [(x,y) for x in xs for y in ys]

#input is the semantically segmented image and the x1,y1, x2,y2  coordinates of bounding box
def semant(frame,xtl,ytl,xbr,ybr):
    #get the two end points
    start= [int(ybr),int(xtl)]
    end = [int(ybr), int(xbr)]
    
    #get points in between
    arr=np.array(points_between(start,end))
    
    #check the mode semantic label
    arr2 =[]
    for i in np.arange(arr.shape[0]):
        arr2.append(frame[arr[i][0],arr[i][1]])
    arr3=np.array(arr2)
    #assign the mode semantic segmented label
    a=stats.mode(arr3)[0][0][0]
    b=stats.mode(arr3)[0][0][1]
    c=stats.mode(arr3)[0][0][2]

    if a==232 and b==35 and c==244:
        return("on pavement")

    elif a==128 and b==64 and c==128:
        return("on road")

    else:
        return("obstructed")
		
		


#input is the sementically segmented image and an nx4 numpy array of x1,y1, x2,y2  coordinates of bounding boxes
def semant_frame(frame,boxes):
    result= []
    
    #boxes[x1,y1,x2,y2]
    for j in np.arange(boxes.shape[0]):
        start= [int(boxes[j][3]),int(boxes[j][0])]
        end = [ int(boxes[j][3]), int(boxes[j][2])]
        arr=np.array(points_between(start,end))
        arr2 =[]
        for i in np.arange(arr.shape[0]):
            arr2.append(frame[arr[i][0],arr[i][1]])
        arr3=np.array(arr2)
        a=stats.mode(arr3)[0][0][0]
        b=stats.mode(arr3)[0][0][1]
        c=stats.mode(arr3)[0][0][2]

        if a==232 and b==35 and c==244:
            result.append("sidewalk")

        elif a==128 and b==64 and c==128:
             result.append("road")

        else:
            result.append("obstructed")
    return result