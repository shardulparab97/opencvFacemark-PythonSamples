#import necessary libraries
import cv2

#load the Cascade classifier
cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt2.xml")

#read the target image
img = cv2.imread("Messi.jpg")

#store the faces detected in the target image
faces = cascade.detectMultiScale(img)

#creating a facemark LBF instance
#see that we are using the cv2.face module
fm = cv2.face.createFacemarkLBF()

#loading the pre-trained model
fm.loadModel("lbfmodel.yaml")

#running the algorithm and storing the landmarks found in the target image
_, landmarks = fm.fit(img,faces)

# print(len(landmarks))

#looping over all the faces obtained and drawing landmarks on them
#uses the drawFacemarks function
for i in range(len(landmarks)):
	cv2.face.drawFacemarks(img,landmarks[i])

#display image
cv2.imshow('Image with landmark detection', img)
cv2.waitKey(0)
cv2.destroyAllWindows()




