from keras.applications.vgg16 import VGG16
model = VGG16 ()
model.summary()

import cv2 
import cv2 as cv
from tkinter import filedialog
from tkinter import*
from matplotlib.widgets import Button
from matplotlib.widgets import TextBox
from matplotlib.widgets import RadioButtons


import requests 
import numpy as np 
import matplotlib.pyplot as plt 
from PIL import Image

fig = plt.figure(figsize= (5, 5))
fig.canvas.manager.set_window_title('Image Prediction (Statistik)')

def insertImage(event): 
    global h, w, img, path
    path= filedialog.askopenfilename()
    img = cv2.imread(path)
    print(path)
    img=cv2.cvtColor(img, cv2.COLOR_BGR2RGBA)
    axImage1.imshow(img)

def reset(event):
    axImage1.clear()
    
def predictImage(event):
    img = cv2.imread(path)
    
    # r=path
    # img = np.array (Image.open (r.raw))
    # plt. subplot(1,len (urls),u+1)
    # plt.imshow(img)
    # img = cv2.resize(img, (224,224)) 
    
    img = cv2.resize(img, (224, 224)) # Resize the image to (224, 224)
    img = np.expand_dims(img, axis=0) # Add a batch dimension of size 1
    print (img.shape) 
    yh = model.predict (img)
    if(np.argmax(yh) >= 147 and np.argmax (yh) <= 299 or np.argmax(yh) >= 330 and np.argmax (yh) <= 388 or np.argmax(yh) >= 147 and np.argmax (yh) <= 299 or np.argmax(yh) >= 103 and np.argmax (yh) <= 106):
        axImage1.set_title("Mammal")
    elif (np.argmax (yh) >= 0 and np.argmax(yh) <= 6 or np.argmax(yh) >= 389 and np.argmax (yh) <= 397):
        axImage1.set_title("Fish")
    elif (np.argmax (yh) >= 7 and np.argmax(yh) <= 24 or np.argmax(yh) >= 127 and np.argmax (yh) <= 146):
        axImage1.set_title("Bird")
    elif (np.argmax (yh) >= 25 and np.argmax(yh) <= 32):
        axImage1.set_title("Amphibia")
    elif (np.argmax (yh) >= 33 and np.argmax(yh) <= 68):
        axImage1.set_title("Reptiles")
    else:
        axImage1.set_title("Probably Not Vertebrate Animal")
            
axImage1 = fig.add_axes([1/8, 1/2, 3/4,5/12])
axPredict = fig.add_axes([0, 1/4, 1, 1/8])
axInsert = fig.add_axes([0, 0, 1/2, 1/4])
axReset = fig.add_axes([1/2, 0, 1/2, 1/4])

buttonPredict = Button(axPredict, 'predict image')
buttonInsert= Button(axInsert, 'insert image')
buttonReset = Button(axReset, 'reset')

buttonPredict.on_clicked(predictImage)
buttonInsert.on_clicked(insertImage)
buttonReset.on_clicked(reset)

plt.show()