import numpy as np 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder, MinMaxScaler 
from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.python.keras import regularizers
from sklearn.metrics import confusion_matrix

# Class distribution: 357 benign, 212 malignant
''' 

Importing data from Breast Cancer Wisconsin (Diagnostic) Data Set Features are
computed from a digitized image of a fine needle aspirate (FNA) of a breast mass. 
They describe characteristics of the cell nuclei present in the image.

'''
dataset = pd.read_csv('data.csv')

# Encoding the data 
dataset_output = dataset.diagnosis
dataset.drop(["diagnosis","Unnamed: 32"],axis = 1 , inplace=True)
dataset.isna().sum()
categorical_cols = dataset.select_dtypes(include=['object']).columns.tolist()
categorical_cols
dataset_output= LabelEncoder().fit_transform(dataset_output,)

# Normalization 
normalizing_scaller =MinMaxScaler()  
normalizing_scaller.fit(dataset)
dataset = normalizing_scaller.transform(dataset) 
 
print(f"Normlized Features its max : \n{dataset.max()} \n\nits min : \n{dataset.min()}" )

# Splitting the dataset into the Training set and Test set
features_train, features_test, output_train, output_test = train_test_split (dataset, dataset_output, test_size = 0.20,random_state =0)


# Initialzation of ANN
model = Sequential()

# Adding the input layer and the first hidden layers
model.add(Dense(120, input_shape=(dataset.shape[1],)))
# Adding dropout to prevent overfitting
model.add(Dropout(0.2))
model.add(Dense(100, activation="relu"))
model.add(Dense(80, activation="relu", kernel_regularizer=regularizers.l2(0.0001) ))
model.add(Dense(80, activation="relu"))
model.add(Dropout(0.2))
model.add(Dense(80, activation="relu", kernel_regularizer=regularizers.l2(0.0001) ))
model.add(Dense(80, activation="relu"))
model.add(Dropout(0.1))
model.add(Dense(20, activation="sigmoid"))

# Adding the output layer
model.add(Dense(1, activation="sigmoid",name= 'Classification_Layer'))

# Model Metric & Optimizer 
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Training 
model.fit(features_train, output_train, batch_size=10, epochs=50, verbose=1)

# Accuracy
test_loss, test_acc = model.evaluate(features_test, output_test)
print(test_acc)

# Predicting the Test set results
y_pred = model.predict(features_test)
y_pred = (y_pred > 0.5)

# Making the Confusion Matrix
cm = confusion_matrix(output_test, y_pred)

print("Our accuracy is {}%".format(((cm[0][0] + cm[1][1])/114)*100))


plt.figure(dpi = 720)
label =['Benign', 'Malignant']
plt.title('Confusion Matrix of the Breast Cancer Classification')

sns.heatmap(cm,annot=True, xticklabels=label,yticklabels=label)
plt.savefig('heatmap.png')
