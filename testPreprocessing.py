import numpy as np #numpy btt3aml m3 arrays
import matplotlib.pyplot as plt #btsa3dny f rsm el plots ll data
import pandas as pd  #btsa3dny a7wl bunch of dataset into a set of table, hanlding popular format of data

train_df = pd.read_csv('Dataset.csv')

X = train_df.iloc[:, :-1].values

from sklearn.impute import SimpleImputer
#hanlding missing values for Gender & Married Objects
imputerObjects = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
imputerObjects.fit(X[:, 1:3])
X[:, 1:3] = imputerObjects.transform(X[:, 1:3])
#print(X[ : ,1:3])

#hanlding missing values for Credit History
imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
X[:, 8:9] = imputer.fit_transform(X[:, 8:9])
#print(X[: , 8:9]) # Credit History

#hanlding missing values for loan & loan amount
imputerNum = SimpleImputer(missing_values=np.nan, strategy='mean')
imputerNum.fit(X[:, 6:8])
X[:, 6:8] = imputerNum.transform(X[:, 6:8])
#print(X[ : ,6:8]) # loan amount $ loan amount term

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
sc.fit(X[:, 3:8])
X[:, 3:8] = sc.transform(X[:, 3:8])
#X[:, 3:8]

from sklearn.preprocessing import MinMaxScaler
mms = MinMaxScaler()
mms.fit(X[:, 3:8])
X[:, 3:8] = mms.transform(X[:, 3:8])
#print(X[: , 3:8])

from sklearn.preprocessing import RobustScaler
rs = RobustScaler()
X[:, 3:8] = rs.fit_transform(X[:, 3:8])
#print (X[:, 3:8])

fig = plt.figure(figsize=(12, 6)) #3shan a7dd m2as elchar
LoanAmount = fig.add_subplot(122)
Loan_Amount_Term = fig.add_subplot(121)

LoanAmount.hist(train_df.LoanAmount, bins=10)
LoanAmount.set_xlabel('LoanAmount')
LoanAmount.set_title("Histogram of LoanAmount")

Loan_Amount_Term.hist(train_df.CoapplicantIncome, bins=10)
Loan_Amount_Term.set_xlabel('Co Applicant income')
Loan_Amount_Term.set_title("Histogram of co applicant Income")
plt.show()

figure = plt.scatter(train_df.CoapplicantIncome, train_df.LoanAmount)
plt.xlabel('co application Income')
plt.ylabel('Loan Amount')
plt.show()

sc_df = pd.DataFrame(X[:, 5:7], columns =['CoapplicantIncome', 'LoanAmount'])
plt.plot(sc_df['CoapplicantIncome'])
plt.plot(sc_df['LoanAmount'])
plt.xlabel('Record Index')
plt.ylabel('StandardScaler Scale w/o outlier')
plt.show()

