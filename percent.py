#!/usr/bin/python

# test file
# TODO:
# 	Figure out four point transform
#	Figure out testing data warping
# 	Use webcam as input
# 	Figure out how to use contours
# 		Currently detects inner rect -> detect outermost rectangle
# 	Try using video stream from android phone

import cv2
import utils as u 
from matplotlib import pyplot as plt
import numpy as np
#import subprocess
#from gtts import gTTS

from tkinter import *
from tkinter import filedialog
root =Tk()
root.title("my project")
# size of the window
root.geometry("500x500")
#set background color
root.config(background="sky blue")

# defining function
def abc():
    #to create a dialogbox to open a file
    filename =filedialog.askopenfilename()
    print(filename)
    return filename

def xyz():
    max_val = 8
    max_pt = -1
    max_kp = 0
    
    orb = cv2.ORB_create()
    # orb is an alternative to SIFT
    
    #test_img = u.read_img('files/test_100_2.jpg')
    #test_img = u.read_img('files/test_50_2.jpg')
    #test_img = u.read_img('files/fn.jpg')
    #test_img = u.read_img('files/test_20_2.jpg')
    #test_img =u.read_img('files/test_100_3.jpg')
    #test_img = u.read_img('files/test_20_4.jpg')
    location = abc()
    #test_img = u.read_img('files/test_20_06.jpg')
    test_img = u.read_img(location)
    #test_img = cv2.resize(Test_img, (1000,1000), interpolation = cv2.INTER_AREA) 
    # resizing must be dynamic
    original = u.resize_img(test_img, 0.4)
    u.display('original', original)
    
    # keypoints and descriptors
    # (kp1, des1) = orb.detectAndCompute(test_img, None)
    (kp1, des1) = orb.detectAndCompute(test_img, None)
    
    training_set = ['files/20.jpg', 'files/50.jpg', 'files/100.jpg', 'files/new100.jpg', 'files/200.jpg', 'files/500.jpg', 'files/2000.jpg']
    
    for i in range(0, len(training_set)):
        #train image
        train_img = cv2.imread(training_set[i])
        (kp2, des2) = orb.detectAndCompute(train_img, None)
        # brute force matcher
        bf = cv2.BFMatcher()    
        all_matches = bf.knnMatch(des1, des2, k=2)
        good = []
        # give an arbitrary number -> 0.789
        # if good -> append to list of good matches
        for (m, n) in all_matches:
            if m.distance < 0.789 * n.distance:
                good.append([m])
                
        
        dist = 1 - len(good) / (max(len(des1), len(des2)))
        
        total = len(kp1) + len(kp2)
        per = len(kp2) / total
        print('percentage', per)
     
        number_keypoint = 0
        if len(kp1) <= len(kp2):
            number_keypoint = len(kp1)
        else:
            number_keypoint = len(kp2)
            
        print('Keypoints of original image: ' +str(len(kp2)))
        print('Keypoints of input image : ' +str(len(kp1)))
        print('Good Matches: ', len(good))
        print('How good is the match: ', len(good) / number_keypoint * 100)
        
        
        if len(good) > max_val:
            max_val = len(good)
            max_pt = i
            max_kp = kp2
    
        print(i, ' ', training_set[i], ' ', len(good))
        print(len(des1),len(des2),dist)
        
       
    if max_val != 8:
        print(training_set[max_pt])
        print('good matches ', max_val)
        
        train_img = cv2.imread(training_set[max_pt])
        img3 = cv2.drawMatchesKnn(test_img, kp1, train_img, max_kp, good, 4)
        note = str(training_set[max_pt])[6:-4]
        print('\nDetected denomination: Rs. ', note)
        #audio_file = 'audio/' + note + '.mp3'
        # audio_file = "value.mp3
        # tts = gTTS(text=speech_out, lang="en")
        # tts.save(audio_file)
        #return_code = subprocess.call(["afplay", audio_file])
        #(plt.imshow(img3), plt.show())
        #cv2.imshow('img3', img3)
        u.display('feature', img3)
        #image=cv2.resize(img3, (700,500), interpolation = cv2.INTER_AREA)
        #cv2.imshow("plot",image)
        #cv2.waitKey()
        #cv2.destroyAllWindows()
    else:
        print('No Matches')
    
    
    
# creating button
btn=Button(root,text="input image",width=50,command=abc)
btn.grid(column=1,row=1)


btn1=Button(root,text="Check",width=50,command=xyz)
btn1.grid(column=1,row=2)

root.mainloop()

