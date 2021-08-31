# Contributed with LUV by JOSHUA JAISTEIN, Academic AI Researcher.

import numpy as np
import cv2
import math

def distanceCalculationcode(l):
    x1,y1 = 0
    l[0] = x1
    l[1] = y1
    x2 = 0
    y2 = 0
    distance = math.sqrt( (((x2) - (x1))**2) + (((y2) - (y1))**2) )
    print( "distance to pedestrian = " + distance)
    return distance

#    MAIN CODE BEGINS
pedestrianCascade = cv2.CascadeClassifier('haarcascade_fullbody.xml')

frame = cv2.imread('pedestrian.jpg')
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
pedestrians = pedestrianCascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(20, 20))

l = []
lf = []

i = 1
s = 'Pedestrians alert !'

# iterate in ROI
for (x,y,w,h) in pedestrians:
    cv2.rectangle(frame, (x,y), (x+w, y+h), (255,0,0),2 )
    cv2.putText(frame,s,(x,y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    l.append(x)
    l.append(y)
    print(l)

print(l)

#distanceCalculationcode(l)

#print(l[0])
#print(l[1])

x1 = 200
y1 = 297

x2 = l[0]
y2 = l[1]

distance = math.sqrt( (((x2) - (x1))**2) + (((y2) - (y1))**2) )
print(distance)


if distance > 200 :
    print('Caution')
else :
    print("Immediate brake initiation")
