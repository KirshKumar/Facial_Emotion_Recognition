import cv2
import pathlib
import numpy as np  
from keras.models import model_from_json  
from tensorflow.keras.utils import load_img, img_to_array
from keras.preprocessing import image 

#load model  
model = model_from_json(open("trained.json", "r").read())  
#load weights  
model.load_weights('trained.h5')
path=pathlib.Path(cv2.__file__).parent.absolute() /"data/haarcascade_frontalface_default.xml"
print(path)

clsfr=cv2.CascadeClassifier(str(path))

cam=cv2.VideoCapture(0)

while 1:
    pic,frame=cam.read()
    if not pic:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    gray_img= cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    facial_coord= clsfr.detectMultiScale(
        gray_img,
        scaleFactor=1.1,
        minNeighbors=7,
        minSize=(48,48),
        flags=cv2.CASCADE_SCALE_IMAGE)



    for (a,b,c,d) in facial_coord:
        cv2.rectangle(frame,(a,b),(a+c,b+d),(255,255,0),2)
        roi_gray=gray_img[b:b+c,a:a+d]#cropping region of interest i.e. face area from  image  
        roi_gray=cv2.resize(roi_gray,(48,48))  
        img_pixels = img_to_array(roi_gray)  
        img_pixels = np.expand_dims(img_pixels, axis = 0)  
        img_pixels /= 255  
  
        predictions = model.predict(img_pixels)  
  
        #find max indexed array  
        max_index = np.argmax(predictions[0])  
  
        emotions = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')  
        predicted_emotion = emotions[max_index]  
  
        cv2.putText(frame, predicted_emotion, (int(a), int(b)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

    resized_frame = cv2.resize(frame, (1000, 700)) 
    cv2.imshow("Face",resized_frame)
    if cv2.waitKey(1)==ord(" "):
        break

cam.release()
cv2.destroyAllWindows()



