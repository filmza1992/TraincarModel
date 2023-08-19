#%%
import cv2
import requests
import numpy as np
import pickle
import os
import base64

from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics

url = "http://localhost:8080/api/genhog"

def img2vec(img):
    v, buffer = cv2.imencode(".jpg", img)
    img_str = base64.b64encode(buffer)
    data = "image data,"+str.split(str(img_str),"'")[1]
    response = requests.get(url, json={"img":data})

    return response.json()["Hog"]

#%%

img_list =[]
path = "Cars Dataset\\train"
i = 0
for subfolder in os.listdir(path):
    for f in os.listdir(os.path.join(path,subfolder)):
        img = cv2.imread(os.path.join(path,subfolder)+"/"+f)
        img22 = []
        img22 = img2vec(img)
        img22.append(i)
        img_list.append(img22) 
    i += 1

#%%
write_path = "ImageFeatureTrain.pkl"
pickle.dump(img_list, open(write_path,"wb"))
print("data preparation is done")

#%%
carVectors = pickle.load(open('ImageFeatureTrain.pkl','rb'))
carVectors_np = np.array(carVectors)
X_train = carVectors_np[:,0:-1]
Y_train = carVectors_np[:,-1]

#%%
path = "Cars Dataset\\test"
img_list =[]
i = 0

for subfolder in os.listdir(path):
    for f in os.listdir(os.path.join(path,subfolder)):
        img = cv2.imread(os.path.join(path,subfolder)+"/"+f)
        
        img22 = img2vec(img)
        img22.append(i)
        img_list.append(img22)
    i += 1

#%%
write_path = "ImageFeatureTest.pkl"
pickle.dump(img_list, open(write_path,"wb"))
print("data preparation is done")

#%%
carVectors = pickle.load(open('ImageFeatureTest.pkl','rb'))
carVectors_np = np.array(carVectors)
X_test = carVectors_np[:,0:-1]
Y_test = carVectors_np[:,-1]

clf = DecisionTreeClassifier()
clf = clf.fit(X_train,Y_train)

y_pred = clf.predict(X_test)



# %%
print("Accuracy:",metrics.accuracy_score(Y_test , y_pred))
# %%
print("Confusion Matrix:", metrics.confusion_matrix(Y_test , y_pred))

# %%
print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)

# %%
path_model = "ImageFeatureModel.pk"
pickle.dump(clf, open(path_model,"wb"))
# %%
 