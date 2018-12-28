import math
from sklearn import neighbors
import os
import cv2
import os.path
import pickle
import numpy as np
from PIL import Image, ImageDraw
from keras.models import load_model
import face_recognition
from face_recognition.face_recognition_cli import image_files_in_folder

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
def train(train_dir, model_save_path=None, n_neighbors=None, knn_algo='ball_tree', verbose=False):
    X = [];y =[]
    for class_dir in os.listdir(train_dir):
        if not os.path.isdir(os.path.join(train_dir, class_dir)):continue
        for img_path in image_files_in_folder(os.path.join(train_dir, class_dir)):
            image = face_recognition.load_image_file(img_path)
            face_bounding_boxes = face_recognition.face_locations(image)

            if len(face_bounding_boxes) != 1:
                if verbose:
                    print("Image {} not suitable for training: {}".format(img_path, "Didn't find a face" if len(face_bounding_boxes) < 1 else "Found more than one face"))
            else:
                X.append(face_recognition.face_encodings(image, known_face_locations=face_bounding_boxes)[0])
                y.append(class_dir)
                
    if n_neighbors is None:
        n_neighbors = int(round(math.sqrt(len(X))))
        if verbose:
            print("Chose n_neighbors automatically:", n_neighbors)

    knn_clf = neighbors.KNeighborsClassifier(n_neighbors=n_neighbors, algorithm=knn_algo, weights='distance')
    knn_clf.fit(X, y)

    if model_save_path is not None:
        with open(model_save_path, 'wb') as f:
            pickle.dump(knn_clf, f)

    return knn_clf

def predict(img, knn_clf=None, model_path=None, distance_threshold=0.55):
    if knn_clf is None and model_path is None:
        raise Exception("Must supply knn classifier either thourgh knn_clf or model_path")
    if knn_clf is None:
        with open(model_path, 'rb') as f:
            knn_clf = pickle.load(f)
    X_face_locations = face_recognition.face_locations(img)
    if len(X_face_locations) == 0:
        return []
    faces_encodings = face_recognition.face_encodings(img, known_face_locations=X_face_locations)
    closest_distances = knn_clf.kneighbors(faces_encodings, n_neighbors=1)
    are_matches = [closest_distances[0][i][0] <= distance_threshold for i in range(len(X_face_locations))]
    return [(pred, loc) if rec else ("unknown", loc) for pred, loc, rec in zip(knn_clf.predict(faces_encodings), X_face_locations, are_matches)]

if __name__ == "__main__":
    print("Training KNN classifier...")
    classifier = train("knn_data/train", model_save_path="trained_knn_model.clf", n_neighbors=2)
    print("Training complete!")
    
    cap = cv2.VideoCapture(0)
    model = load_model('experssion_detector.hdf5')
    target = ['very bad','disgusting','worse','very good','bad','good','neutral']
    review=[]
    font=cv2.FONT_HERSHEY_SIMPLEX
    while 1:
        namw1=''
        ret, img = cap.read()
        predictions = predict(img, model_path="trained_knn_model.clf")
        for name, (h, x, y, w) in predictions:
            name1=name
            cv2.rectangle(img,(w,h),(x,y),(255,0,0),2)
            face_crop = img[h:y,w:x]
            face_crop = cv2.resize(face_crop,(48,48))
            face_crop = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
            face_crop = face_crop.astype('float32')/255
            face_crop = np.asarray(face_crop)
            face_crop = face_crop.reshape(1, 1,face_crop.shape[0],face_crop.shape[1])
            result = target[np.argmax(model.predict(face_crop))]
            review.append(result)
            cv2.putText(img, str(name)+" ("+result+")",(w,y),font,1,(255,255,0),2)

        if len(review)>=21:
            print(name1+"'s review is "+max(set(review), key=review.count))
            break    
        cv2.imshow("Face recognition",img)
        if cv2.waitKey(1) & 0xFF == ord('q'):break
    cap.release()
    cv2.destroyAllWindows()
