import numpy as np  #numpy btt3aml m3 arrays
import matplotlib.pyplot as plt #btsa3dny f rsm el plots ll data
import pandas as pd  #btsa3dny a7wl bunch of dataset into a set of table, hanlding popular format of data
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
import seaborn as sns
from sklearn.compose import ColumnTransformer

dataset = pd.read_csv('Dataset.csv')

X = dataset.iloc[:,:-1].values
y = dataset.iloc[:, -1].values

# hanlding missing values for categories
imputerObjects = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
X[:, 1:4] = imputerObjects.fit_transform(X[:, 1:4])  # married , gender , dependent
X[:, 8:9] = imputerObjects.fit_transform(X[:, 8:9]) # property area
#print(X[:,1:4])

# hanlding missing values for loan & loan amount
imputerNum = SimpleImputer(missing_values=np.nan, strategy='mean')
imputerNum.fit(X[:, 6:8])
X[:, 6:8] = imputerNum.transform(X[:, 6:8])
#print(X[ : ,6:8])

le = LabelEncoder()
for i in range(3): 
    X[:, i] = le.fit_transform(X[:, i])
X[:,9] = le.fit_transform(X[:,9]) # encode property area

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

from  sklearn.preprocessing import StandardScaler
sc = StandardScaler()
sc.fit(X_train)
X_train = sc.transform(X_train)
X_test= sc.transform(X_test)


from sklearn.neighbors import KNeighborsClassifier
Classifier = KNeighborsClassifier(n_neighbors = 7)
Classifier.fit(X_train , y_train)
y_pred = Classifier.predict(X_test)
print(y_pred)

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
print ('Accuracy: ', accuracy_score(y_test,y_pred))
print ("\n Classification Report: \n",classification_report(y_test , y_pred))
cm = confusion_matrix(y_test, y_pred)
print ('\nConfusion Matrix: \n', cm)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", linewidths=.5, cbar=False)
plt.title('Confusion Matrix')
plt.xlabel('Predicted Labels')
plt.ylabel('Actual Values')
plt.show()


#sc.transform([['LP001030','Male','Yes','1',2666,500,17,360,1,'Urban']])

# Define the input data , list of list to represent row data
data = [['1',	'Female',	'Yes',	'1',	0	,13	,10,	360,	0	,'Rural']]

# Define the indices of the columns containing string values
string_columns = [0, 1, 2, 9]

# Convert string columns to numerical labels
for col in string_columns:
    data[0][col] = le.fit_transform([data[0][col]])[0]

# Print the transformed data
print(data)

new_pred = Classifier.predict(sc.transform(data))
print(new_pred)

print(Classifier.predict(sc.transform([['1','1', '1',	'1',	381	,13	,10,	360,	0 ,'1']])))