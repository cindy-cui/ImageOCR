# import os 
# os.environ['KMP_DUPLICATE_LIB_OK']='True'
import easyocr
import cv2
from PIL import Image
import numpy as np
from scipy import ndimage
import time
import sys


def comparator(file,file_1):   
    file=open(file,'r')
    file_1=open(file_1,'r')
    binary_1=list(file.read())
    binary=list(file_1.read())
    file_1.close()
    file.close()
    i=0
    for x,y in zip(binary,binary_1):
        if(x==y):
            i=i+1
        else:
            continue
    result=(i/len(binary))*100
    print('the Match is {0}%'.format(result))
        


langs = ["en"]
print("[INFO] OCR'ing with the following languages: {}".format(langs))
# load the input image
image = cv2.imread(str(sys.argv[1]))

#convert image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Resizing to > 300 dpi; increasing scanning resolution 
x, y = gray.shape
factor = min(1, float(1024.0 / x))
size = (int(factor * y), int(factor * x))
image = cv2.resize(image, size)
gray = cv2.resize(gray, size)

gray2 = gray.copy() #replicate the image



#method 1 with morphological transformation
# Morph open to remove noise
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2)) #cpytreate a rectangular kernel 
#convolution matrix 
opening = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel, iterations=1) #use morphological operations to remove noise 
# Find contours and remove small noise
cnts = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if len(cnts) == 2 else cnts[1]
for c in cnts:
    area = cv2.contourArea(c)
    if area < 50:
       cv2.drawContours(opening, [c], -1, 0, -1) #fill in contours 
# Invert and apply slight Gaussian blur
result = 255 - opening
result = cv2.GaussianBlur(result, (3,3), 0)
gray = result 



#method 2 with thresholding/denoising
gray2 = cv2.adaptiveThreshold(gray2,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,33,11) #thresholding for uneven lighting
gray2 = cv2.fastNlMeansDenoising(gray2,None,10,33,11) #gaussian white noise





# OCR the input image using EasyOCR
print("[INFO] OCR'ing input image...")
reader = easyocr.Reader(langs, gpu=True)
results = reader.readtext(gray)
results2 = reader.readtext(gray2)


file1 = open("testerphotoMORPH.txt", "w")
for (bbox, text, prob) in results:
    file1.write(text) # write the ocr-ed text to the file
    file1.write("\n")
file1.close()


file2 = open("testerphotoNOISE.txt", "w")
for (bbox, text, prob) in results2:
    file2.write(text)
    file2.write("\n")
file2.close()


comparator('testerphoto.txt','testerphotoMORPH.txt')
comparator('testerphoto.txt','testerphotoNOISE.txt')


cv2.waitKey(0)