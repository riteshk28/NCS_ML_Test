


print('importing required libraries')

import pandas as pd
import numpy
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.preprocessing import sequence

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn import preprocessing


#importing dataset

print('importing and processing dataset')
df = pd.read_csv("name_gender.csv", header=None)


#Naming columns, converting all letters in lowercase
#and only choosing the names which are not more than 10 character labels
df = df.rename(columns={0:'Name',1:'Gender',2:'Other'})
df = df.apply(lambda x: x.astype(str).str.lower())
short_df = df[df.Name.str.len()<=10]

#Replacing male and female label with 0 and 1 respectively.
short_df['Gender'] = short_df['Gender'].replace(['m'],int(0))
short_df['Gender'] = short_df['Gender'].replace(['f'],int(1))



# We will now create 10 columns, each column is a representaion of each letter's index position in English alphabets. If the name has less than 10 letters, the rest of the column will be set to 0.(we will not have more than 10 letters since we have already filtered it above.
# 
# For example,
# Name 'ABC' becomes 1 2 3 0 0 0 0 0 0 0
# 


#creating an empty dictionary for appending later.
column = ['firstchar','secondchar','thirdchar','fourthchar','fifthchar','sixthchar','seventhchar','eighthchar','ninthchar','tenthchar']
dict = {new_list:[] for new_list in column}



#append the first column 'firstchar' with the alphabetical number of the respective name's first letter.
#We repeat this 10 times. (There could be a faster and efficient way to write this)
for i in short_df.Name:
    try:
        dict['firstchar'].append(ord(i[0])-96)
    except IndexError as I:
        dict['firstchar'].append(0)




for i in short_df.Name:
    try:
        dict['secondchar'].append(ord(i[1])-96)
    except IndexError as I:
        dict['secondchar'].append(0)




for i in short_df.Name:
    try:
        dict['thirdchar'].append(ord(i[2])-96)
    except IndexError as I:
        dict['thirdchar'].append(0)


for i in short_df.Name:
    try:
        dict['fourthchar'].append(ord(i[3])-96)
    except IndexError as I:
        dict['fourthchar'].append(0)


for i in short_df.Name:
    try:
        dict['fifthchar'].append(ord(i[4])-96)
    except IndexError as I:
        dict['fifthchar'].append(0)



for i in short_df.Name:
    try:
        dict['sixthchar'].append(ord(i[5])-96)
    except IndexError as I:
        dict['sixthchar'].append(0)



for i in short_df.Name:
    try:
        dict['seventhchar'].append(ord(i[6])-96)
    except IndexError as I:
        dict['seventhchar'].append(0)



for i in short_df.Name:
    try:
        dict['eighthchar'].append(ord(i[7])-96)
    except IndexError as I:
        dict['eighthchar'].append(0)



for i in short_df.Name:
    try:
        dict['ninthchar'].append(ord(i[8])-96)
    except IndexError as I:
        dict['ninthchar'].append(0)




for i in short_df.Name:
    try:
        dict['tenthchar'].append(ord(i[9])-96)
    except IndexError as I:
        dict['tenthchar'].append(0)




#Converting the dict in pandas dataframe
char_df = pd.DataFrame(dict)



#attach the target variable with char_df and dropping NAs

final_df = pd.concat([char_df, short_df.Gender],axis=1)
final_df = final_df.dropna()



#convert df in tyoe int for simplicity
final_df = final_df.astype(int)



#Define X, y, train and test set
#scale the features with StandardScalar()

X = final_df.drop(['Gender'], axis=1)
y = final_df['Gender']

scaler = preprocessing.StandardScaler().fit(X)
scaled_train = scaler.transform(X)
X_train, X_test, y_train, y_test = train_test_split(scaled_train, y, test_size=0.20, random_state=42)

print('train and test set created')




# define the keras model
model = Sequential()
model.add(Dense(12, input_dim=10, activation='relu'))
model.add(Dense(8, activation='relu', use_bias=True, bias_initializer='zeros'))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))



# compile the keras model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])




print('Model training in process...')
# fit the keras model on the dataset
model.fit(X_train, y_train, epochs=50, batch_size=10)





# serialize model to JSON
model_json = model.to_json()
with open("predict_gender.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("predict_gender.h5")


print("Training complete. Model saved to current working directory")


predicted = model.predict_classes(X_test)



print('accuracy on test set is : ' + str(accuracy_score(y_test, predicted)))

