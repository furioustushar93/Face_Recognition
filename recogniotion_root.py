import numpy as np
import cv2
import os


def distance(x1, x2):
    return np.sqrt(sum((x1 - x2) ** 2))


def knn(train, test, k=5):
    dist = []

    for i in range(train.shape[0]):
        # Get the vector and label
        ix = train[i, :-1]
        iy = train[i, -1]
        # Compute the distance from test point
        d = distance(test, ix)
        dist.append([d, iy])
    # Sort based on distance and get top k
    dk = sorted(dist, key=lambda x: x[0])[:k]
    # Retrieve only the labels
    labels = np.array(dk)[:, -1]

    # Get frequencies of each label
    output = np.unique(labels, return_counts=True)
    # Find max frequency and corresponding label
    index = np.argmax(output[1])
    return output[0][index]


cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")

skip = 0
data_path = './data/'
face_data = []
face_section = 0
class_id = 0
label = []
names = {}

#Data Preparation
for fx in os.listdir(data_path):
    if fx.endswith('.npy'):
        names[class_id] = fx[:-4]
        dataitem = np.load(data_path+fx)
        face_data.append(dataitem)

        target = class_id*np.ones((dataitem.shape[0],))
        label.append(target)


face_dataset = np.concatenate(face_data, axis = 0)
face_label = np.concatenate(label, axis = 0).reshape((-1,1))

train_set = np.concatenate((face_dataset,face_label),axis = 1)


while True:

    ret,frame = cap.read()

    if ret == False:
        continue
    faces  =  face_cascade.detectMultiScale(frame,1.3,5)

    for face in faces:

        x,y,w,h = face

        offset = 10
        face_section = frame[y-offset:y+h+offset, x-offset:x+w+offset]
        face_section = cv2.resize(face_section,(100,100))

        out = knn(train_set,face_section.flatten())

        pred_name = names[int(out)]
        cv2.putText(frame,pred_name,(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2,cv2.LINE_AA)
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),2)
    cv2.imshow("faces",frame)

    key_pressed = cv2.waitKey(1) & 0xFF
    if key_pressed == ord('c'):
        break
cap.release()
cv2.destroyAllWindows()