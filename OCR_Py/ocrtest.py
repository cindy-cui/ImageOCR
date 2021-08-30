# import os 
# os.environ['KMP_DUPLICATE_LIB_OK']='True'
import easyocr
import cv2
from PIL import Image
import numpy as np
from scipy import ndimage
import time
import sys

start = time.time()

langs = ["en"]
print("[INFO] OCR'ing with the following languages: {}".format(langs))
# load the input image
image = cv2.imread(str(sys.argv[1]))

#convert image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)



# gray = cv2.GaussianBlur(gray, (5, 5), 0) #reduces image noise 




# Resizing to > 300 dpi; increasing scanning resolution 
x, y = gray.shape
factor = min(1, float(1024.0 / x))
size = (int(factor * y), int(factor * x))
image = cv2.resize(image, size)
gray = cv2.resize(gray, size)

# Autorotation
# img_edges = cv2.Canny(gray, 100, 100, apertureSize=3)
# lines = cv2.HoughLinesP(img_edges, 1, math.pi / 180.0, 100, minLineLength=100, maxLineGap=5)

# angles = []
# if lines is not None and lines.shape[0] > 0:
#     for [[x1, y1, x2, y2]] in lines:
#         cv2.line(image, (x1, y1), (x2, y2), (255, 0, 0), 3)
#         angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
#         angles.append(angle)

#     cv2.imshow("Detected lines", image) 

#     median_angle = np.median(angles)
#     gray = ndimage.rotate(gray, median_angle)
#     image = ndimage.rotate(image, median_angle)
# else:
#     print("no lines detected")

# check to see if we should apply thresholding to preprocess the
# image


# Morph open to remove noise
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2)) #create a rectangular kernel 
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




#gray = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,33,11) #thresholding for uneven lighting
#gray = cv2.fastNlMeansDenoising(gray,None,10,33,11) #gaussian white noise






# OCR the input image using EasyOCR
print("[INFO] OCR'ing input image...")
reader = easyocr.Reader(langs, gpu=True)
results = reader.readtext(gray)

# loop over the results
for (bbox, text, prob) in results:
    # display the OCR'd text and associated probability
    print("[INFO] {:.4f}: {}".format(prob, text))
    # unpack the bounding box
    (tl, tr, br, bl) = bbox
    tl = (int(tl[0]), int(tl[1]))
    tr = (int(tr[0]), int(tr[1]))
    br = (int(br[0]), int(br[1]))
    bl = (int(bl[0]), int(bl[1]))
    # cleanup the text and draw the box surrounding the text along
    # with the OCR'd text itself
#     text = cleanup_text(text)

    cv2.rectangle(image, tl, br, (0, 0, 255), 2)
    cv2.putText(image, text, (tl[0], tl[1] - 10),
        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
# show the output image
end = time.time()
print("time elasped: " + str(end - start))
cv2.imshow("Gray", gray)
cv2.imshow("Image", image)
cv2.waitKey(0)