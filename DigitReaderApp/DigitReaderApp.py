import numpy as nm
import tensorflow as tf
import matplotlib.pyplot as plt

print(tf.__version__)
print(nm.__version__)

import sys
print(sys.executable)

import cv2

print("CV2 version " + cv2.__version__)



digit_model = tf.keras.models.load_model("digit-recognizer.keras")

X_testReload = nm.loadtxt("X_testreshaped.csv")
Y_testReload = nm.loadtxt("Y_testreshaped.csv")
X_testFrames = nm.reshape(X_testReload,(10000, 28, 28));
Y_testFrames = nm.reshape(Y_testReload,(10000));

print(X_testFrames[0].shape)
print(Y_testFrames.shape)

print("expected result")  
print(Y_testFrames[0])

input_frame = nm.reshape(X_testFrames[0], (1, 28, 28))

result = digit_model.predict(input_frame)

print("actual result") 
print(result)



#plt.imshow(X_testFrames[0],cmap='gray', vmin=0, vmax=255)
#plt.show()





cv2.namedWindow("Camera")
vc = cv2.VideoCapture(0)


if vc.isOpened(): # try to get the first frame
    rval, frame = vc.read()
else:
    rval = False

while rval:
    cv2.imshow("Camera", frame)
    rval, frame = vc.read()
    key = cv2.waitKey(20)
   
    if key == 27: # exit on ESC
        break
    elif key == 99:
        print ("frame captured")
        #cv2.imshow("Captured Frame", frame)
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        resized_frame = cv2.resize(gray_frame, (28, 28))
        #resized_frame = gray_frame
        
        
        #cv2.imshow("Captured Frame", resized_frame)
        cv2.imshow("Captured Frame", gray_frame)
        retVal, inverted_threshold_frame = cv2.threshold(resized_frame,100,255,cv2.THRESH_BINARY_INV)
        #retVal, inverted_threshold_frame = cv2.threshold(gray_frame,0,100,cv2.THRESH_BINARY)

        #cv2.imshow("Captured Frame", inverted_threshold_frame)
        
        #inverted_frame = 255 - resized_frame
        #X_captured = nm.array(resized_frame)
        X_captured = nm.array(inverted_threshold_frame)
        #print(X_captured.shape)
        input_frame = nm.reshape(X_captured, (1, 28, 28))
        
        
        #print(input_frame.shape)
        result = digit_model.predict(input_frame)
        print("actual result") 
        print(result)

        
        plt.imshow(X_captured,cmap='gray', vmin=0, vmax=255)
        plt.show()
        
vc.release()
cv2.destroyWindow("Camera")