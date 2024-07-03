import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import precision_score, recall_score, f1_score
from matplotlib.pyplot import imshow
import random

# Coding one hot labels
encoder = OneHotEncoder()
encoder.fit([[0], [1]])

# This cell updates result list for images with tumor
data = []
paths = []
result = []

for r, d, f in os.walk(r'brain_tumor_dataset/yes'):
    for file in f:
        if '.jpg' in file:
            paths.append(os.path.join(r, file))


for path in paths:
    img = Image.open(path)
    img: Image = img.resize((128, 128))
    img = np.array(img)
    if img.shape == (128, 128, 3):
        data.append(np.array(img))
        result.append(encoder.transform([[0]]).toarray())


# This cell updates result list for images without tumor

paths = []
for r, d, f in os.walk(r"brain_tumor_dataset/no"):
    for file in f:
        if '.jpg' in file:
            paths.append(os.path.join(r, file))


for path in paths:
    img = Image.open(path)
    img = img.resize((128, 128))
    img = np.array(img)
    if img.shape == (128, 128, 3):
        data.append(np.array(img))
        result.append(encoder.transform([[1]]).toarray())

data = np.array(data)
result = np.array(result)
result = result.reshape(139, 2)

# Splitting data for test and train sets
x_train, x_test, y_train, y_test = train_test_split(data, result, test_size=0.2, shuffle=True, random_state=0)

# Model building ==>
model = Sequential()

model.add(Conv2D(32, kernel_size=(2, 2), input_shape=(128, 128, 3), padding = 'Same'))
model.add(Conv2D(32, kernel_size=(2, 2),  activation ='relu', padding = 'Same'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(64, kernel_size=(2, 2), activation='relu', padding='Same'))
model.add(Conv2D(64, kernel_size=(2, 2), activation='relu', padding='Same'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(2, activation='softmax'))

model.compile(loss="categorical_crossentropy",
              optimizer="Adamax",
              metrics=["accuracy"])

history = model.fit(x_train, y_train, epochs=30, batch_size=40, verbose=1, validation_data=(x_test, y_test))

# Calculation metrics ==>
evaluation_results = model.evaluate(x_test, y_test)

# 'Evaluate' method returns a list of metrics, including accuracy
accuracy = evaluation_results[1]
y_pred = model.predict(x_test)

# Convert probabilities to class labels (assuming binary classification)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = np.argmax(y_test, axis=1)

# Calculate precision, recall, and F1 score
precision = precision_score(y_true_classes, y_pred_classes)
recall = recall_score(y_true_classes, y_pred_classes)
f1 = f1_score(y_true_classes, y_pred_classes)

# Printing metrics
print("Model Accuracy on Test Set: {:.2f}%".format(accuracy * 100))
print("Precision: {:.2f}".format(precision))
print("Recall: {:.2f}".format(recall))
print("F1 Score: {:.2f}".format(f1))

# Creating plots ==>
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Test', 'Validation'], loc='upper right')
plt.grid()
plt.show()


# Checking the model for correct operation
def names(number):
    if number == 0:
        return 'the brain with a tumor'
    else:
        return 'the brain without a tumor'


# Without tumor
no_folder = "brain_tumor_dataset/no"
file_names_n = os.listdir(no_folder)
random_file_name_n = random.choice(file_names_n)
print("Chosen file name is: ", random_file_name_n)
img_path = os.path.join(no_folder, random_file_name_n)
img = Image.open(img_path)
img.show()

x = np.array(img.resize((128, 128)))
x = x.reshape(1, 128, 128, 3)
res = model.predict_on_batch(x)
classification = np.where(res == np.amax(res))[1][0]
imshow(img)
print(str(res[0][classification]*100) + '% confidence, that ' + names(classification))

# With tumor
yes_folder = "brain_tumor_dataset/yes"
file_names_y = os.listdir(yes_folder)
random_file_name_y = random.choice(file_names_y)
print("Chosen file name is: ", random_file_name_y)
img_path_2 = os.path.join(yes_folder, random_file_name_y)
img = Image.open(img_path_2)
img.show()

x = np.array(img.resize((128, 128)))
x = x.reshape(1, 128, 128, 3)
res = model.predict_on_batch(x)
classification = np.where(res == np.amax(res))[1][0]
imshow(img)
print(str(res[0][classification]*100) + '% confidence, that ' + names(classification))

# With all data
folder = "brain_tumor_dataset/photos_with_yes_and_no"
file_names = os.listdir(folder)
random_file_name = random.choice(file_names)
print("Chosen file name is: ", random_file_name)
img_path = os.path.join(folder, random_file_name)
img = Image.open(img_path)
img.show()

x = np.array(img.resize((128, 128)))
x = x.reshape(1, 128, 128, 3)
res = model.predict_on_batch(x)
classification = np.where(res == np.amax(res))[1][0]
imshow(img)
print(str(res[0][classification]*100) + '% confidence, that ' + names(classification))