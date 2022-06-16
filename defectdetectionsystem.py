import os
import cv2
import numpy as np
from tkinter import *
from PIL import Image
from PIL import ImageTk
import tkinter.filedialog



################################
# --- Variables ---
################################
IMAGE_SIZE = (500, 500)

# -- Threshold Details --
# *THRESHOLD_VALUE need to INCREASED if no contour detected,
# *if there're INACCURATE CONTOUR NUMBERS the value need to DECREASED
THRESHOLD_VALUE = 110
MAX_VALUE = 255

# -- Invert Threshold Details --
INV_THRESHOLD_VALUE = 50
INV_MAX_VALUE = 255

# -- Canny Details --
THRESHOLD1 = 100
THRESHOLD2 = 70

# --contour properties--
CON_COLOR = (0, 0, 255)
CON_THICKNESS = 1

# -- Image Stack properties--
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREEN = (0, 255, 0)
RED = (0, 0, 255)
STACK_IMG_SIZE = (200, 200)

root= tkinter.Tk()

canvas1 = tkinter.Canvas(root, width = 300, height = 300)
canvas1.pack()

def select_image():
    label1 = tkinter.Label(root, text= 'Hello World!', fg='green', font=('helvetica', 12, 'bold'))
    canvas1.create_window(150, 200, window=label1)
    
# grab a reference to the image panels
    global panelA, panelB
button1 = tkinter.Button(text='Click Me',command=select_image, bg='brown',fg='white')
canvas1.create_window(150, 150, window=button1)

# open a file chooser dialog and allow the user to select an input
# image
path = tkinter.filedialog.askopenfilename()
# Image Path
imageOri = cv2.imread(path)

        # converts to grayscale
image = cv2.cvtColor(imageOri, cv2.COLOR_BGR2GRAY)


    # resize image
    
image = cv2.resize(image, IMAGE_SIZE)
imageOri = cv2.resize(imageOri, IMAGE_SIZE)
image = cv2.GaussianBlur(image, (3, 3), 0)

# Threshold the image so that your black markings are black on a white background.
ret, thresh_basic = cv2.threshold(image, THRESHOLD_VALUE, MAX_VALUE, cv2.THRESH_BINARY)

    # show thresholded image - DEBUGGING
    #imcv2.imshow("Thresh basic", thresh_basic)

    
    # cv2.imshow("Thresh Adapt", thresh_addapt)

    # Taking a matrix of size 5 as the kernel
kernel = np.ones((5, 5), np.uint8)
 
    # Morphological operations-Erodes away the boundaries of foreground object
    # Use morphology to clean up extraneous markings.
img_erosion = cv2.erode(thresh_basic, kernel, iterations=1)

    #####################
    # The invert the thresholded image,
    # so that the black markings are white on a black background and then find the external contours of those.
ret, thresh_inv = cv2.threshold(img_erosion, INV_THRESHOLD_VALUE, INV_MAX_VALUE, cv2.THRESH_BINARY_INV)
    # show inverted threshold image - DEBUGGING
    #cv2.imshow("INV", thresh_inv)


    #####################

    # Find Canny edges
edged = cv2.Canny(img_erosion, THRESHOLD1, THRESHOLD2)
    # show canny edges - DEBUGGING
    #cv2.imshow('Canny', edged)
    #cv2.waitKey(0)

    # Find Contours
    # findContours alters the image
contours, hierarchy = cv2.findContours(thresh_inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # ++++++++++++++++++
    # -- Image Stack  --
    # ++++++++++++++++++
font = cv2.FONT_HERSHEY_SIMPLEX

imageRz = cv2.resize(image, STACK_IMG_SIZE)
thresh_basicRz = cv2.resize(thresh_basic, STACK_IMG_SIZE)
img_erosionRz = cv2.resize(img_erosion, STACK_IMG_SIZE)
thresh_invRz = cv2.resize(thresh_inv, STACK_IMG_SIZE)
edgedRz = cv2.resize(edged, STACK_IMG_SIZE)

imageRz = cv2.putText(imageRz, 'GrayScale', (5, 15), font, 0.5, WHITE, 1, cv2.LINE_AA)
thresh_basicRz = cv2.putText(thresh_basicRz, 'ThresholdBasic', (5, 15), font,
                                 0.5, WHITE, 1,cv2.LINE_AA)
img_erosionRz = cv2.putText(img_erosionRz, 'Morphology-Erosion', (5, 15), font,
                                0.5, WHITE, 1, cv2.LINE_AA)
thresh_invRz = cv2.putText(thresh_invRz, 'Threshold-mode INV', (5, 15), font,
                               0.5, BLACK, 1, cv2.LINE_AA)
edgedRz = cv2.putText(edgedRz, 'Canny Edges', (5, 15), font, 0.5, WHITE, 1, cv2.LINE_AA)

numpy_horizontal_concat = np.concatenate((imageRz, thresh_basicRz, img_erosionRz,
                                                  thresh_invRz, edgedRz), axis=1)

#cv2.imshow('Filtering...', numpy_horizontal_concat)

    # +++++++

    # get total contours
num_of_con = str(len(contours) - 1)
print("Number of Contours found = " + num_of_con)
if len(contours) > 1:
    print('======================================')
    print('=       MARKINGS DETECTED            =')
    print('======================================\n\n')

    # show original img

cv2.imshow('Original Image', imageOri)
    # draw contours on original img
if int(num_of_con) != 0:
    for i in range(int(num_of_con)):
        highlighted_img = cv2.drawContours(imageOri, contours, i, CON_COLOR, CON_THICKNESS)

        highlighted_img = cv2.putText(highlighted_img, 'Approximately {} defect(s) detected'.
                                      format(num_of_con), (5, 15),
                                      font, 0.5, GREEN, 1, cv2.LINE_AA)
else:
    highlighted_img = cv2.putText(imageOri, 'Unable to detect defects!',
                                          (5, 15), font, 0.5, RED, 2, cv2.LINE_AA)

# show markings highlighted img
cv2.imshow('Highlighted Defect', highlighted_img)
# save image containing highlighted defect
cv2.imwrite('Output Images/{}_DEFECTS_HIGHLIGHTED.jpg'.format(path.split('.')[0]), highlighted_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
