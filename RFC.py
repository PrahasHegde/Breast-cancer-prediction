#USING RANDOM FOREST CLASSIFIER (97% accuracy)

#imports
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score,confusion_matrix
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier


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

#Random Forest Classifier
rfc = RandomForestClassifier()
rfc.fit(X_train, y_train)
rfc_prediction = rfc.predict(X_test)

#classification report
print(classification_report(y_test, rfc_prediction))

#metrics
print(accuracy_score(y_test, rfc_prediction) ) # 0.9736842105263158

#confusion matrix of actual vs predicted
confmat = confusion_matrix(y_test, rfc_prediction)
print(confmat)

sns.heatmap(confmat, annot=True)
plt.title('confusion matrix')
plt.show()
