#USING DECISION TREE CLASSIFIER (92% accuracy)

#imports
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score,confusion_matrix
import plotly.express as px
from sklearn.tree import DecisionTreeClassifier


#Load dataset
pd.set_option('display.max_columns', 100)
df = pd.read_csv('breast-cancer.csv')

print(df.head())
print(df.shape)
print(df.info())
print(df.isnull().sum())

#label column
print(df['diagnosis'].value_counts()) 
#drop id column
df.drop(columns='id', inplace=True)

#handling categorical values from daignsis column
le = LabelEncoder()
df['diagnosis'] = le.fit_transform(df['diagnosis'])
print(df.head())


#split the dataset into features and labels
X = df.drop(columns=['diagnosis'])
y = df['diagnosis']

#train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=786)
print(X_train.shape, X_test.shape) # ((455, 30), (114, 30))
print(y_train.shape, y_test.shape) # ((455,), (114,))

#Decision Tree Classifier
dtc = DecisionTreeClassifier()
dtc.fit(X_train, y_train)
dtc_prediction = dtc.predict(X_test)

#classification report
print(classification_report(y_test, dtc_prediction))

#metrics
print(accuracy_score(y_test, dtc_prediction) ) # 0.9298245614035088

#confusion matrix of actual vs predicted
confmat = confusion_matrix(y_test, dtc_prediction)
print(confmat)

sns.heatmap(confmat, annot=True)
plt.title('confusion matrix')
plt.show()
