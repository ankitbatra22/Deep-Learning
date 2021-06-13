import cv2
import os
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt


### PREPROCESSING ###

# in preprocessing, don't wanna build the data everytime. Set to true first time. 
REBUILD = False

class Recognize():
  # All images are different sizes so we need to normalize to 50x50
  SIZE = 50
  CATS = "PetImages/Cat"
  DOGS = "PetImages/Dog"
  LABELS = {CATS: 0, DOGS: 1}
  training_data = []
  
  catCount = 0
  dogCount = 0

  def makeTrainingData(self):
    #iterate over directories
    for label in self.LABELS:
      print(label)
      #iterate over the images in the directory
      for f in tqdm(os.listdir(label)):
        try: 
          # f is the filename, so join label with filename
          path = os.path.join(label, f)
          # convert image to grayscale
          img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
          # resize
          img = cv2.resize(img, (self.SIZE, self.SIZE))
          #image followed by one hot vector using np.eye
          #self.training_data.append([np.array(img), np.eye(2)[self.LABELS[label]]])
          self.training_data.append([np.array(img), self.LABELS[label]])
          #Counter
          if label == self.CATS:
            self.catCount += 1
          elif label == self.DOGS:
            self.dogCount += 1
        except Exception as e:
          pass 

        
        np.random.shuffle(self.training_data)
        np.save("training_data.npy", self.training_data)

        print("CATS:", self.catCount)
        print("DOGS:", self.dogCount)

if REBUILD:
  model = Recognize()
  model.makeTrainingData()

# Make Training Data
training_data = np.load("training_data.npy", allow_pickle=True)
print(len(training_data))
#print((training_data).shape)
#print(training_data[0:200])
#print(type(training_data[0][1]))

"""
print(training_data[13000][0].shape)
plt.imshow(training_data[13000][0].reshape(50,50,1))
plt.show()


print(training_data[13000])"""
#print(training_data[200][1])

fig = plt.figure(figsize=(25, 4))
for i in np.arange(20):
    ax = fig.add_subplot(2, 20/2, i+1, xticks=[], yticks=[])
    ax.imshow(np.squeeze(training_data[i][0]))
    #ax.set_title("{} ({})".format(str(training_data[i][1])))
    #ax.set_title("testing",training_data[i][1])
    #ax.set_title(str(training_data[i][1]))
    #ax.set_title(("{} ({})".format(str(training_data[i][0]))))
    ax.set_title(training_data[i][1])

plt.show()



