import pandas as pd
import numpy as np
df = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/Iris.csv')
df.head(5)
df.shape
df.describe()
x = df['Species']
print(x)
df.rename(columns={"SepalLengthCm":"SLC","SepalWidthCm":"SWC","PetalLengthCm":"PLC","PetalWidthCm":"PWC"}).head()
x = df["SepalWidthCm"]
print(x.head())
y = df['Species']
x = df.drop('Species', axis = 1)
#x = x.drop("id", axis = 1)
from sklearn.model_selection  import train_test_split
X_train,X_test,y_train,y_test = train_test_split(x,y,test_size = 0.3,random_state = 0)
from sklearn.neighbors import KNeighborsClassifier


K = []
training = []
test = []
scores = {}

for k in range(11, 81):
    clf = KNeighborsClassifier(n_neighbors = k)
    clf.fit(X_train, y_train)

    training_score = clf.score(X_train, y_train)
    test_score = clf.score(X_test, y_test)
    K.append(k)

    training.append(training_score)
    test.append(test_score)
    scores[k] = [training_score, test_score]
  for keys, values in scores.items():
    print(keys, ':', values)
    y_pred = clf.predict(X_test)
    from sklearn.metrics import classification_report, confusion_matrix,ConfusionMatrixDisplay
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
cm = confusion_matrix(y_test, y_pred, labels=clf.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=clf.classes_)
disp.plot()
