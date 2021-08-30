# Installing EasyOCR
EasyOCR can be installed through pip by running the following command:
`pip install -r requirements.txt`
If running on Windows, make sure that torch and torchvision (of appropriate versions) are already installed. They can be installed by following the official instruction on https://pytorch.org. If you wish to run EasyOCR with GPU, be sure to select CUDA 10.2 or CUDA 11.1 (depending on your GPU card) under Compute Platform when installing torch.
After intalling EasyOCR, please make sure that the version you're using is newer than 1.2. Earlier versions might a bug that keeps model downloading from happening.

# Running the script
`fulltest.py` runs Tesseract and EasyOCR on the input image, and compares their output and runtime. It takes three parameters: The first one is the number of runs to perform on the image (int), the second one is a boolean indicating whether GPU should be used for EasyOCR (bool), and the third one is the name of the image file (str). If I am to run the script with the image `test.png` while deactivating CUDA, and have both algorithms run 100 times, I would enter the following command:
`python fulltest.py 100 0 test.png`

`ocrtest.py` runs EasyOCR on the input image and outputs the result. It takes one parameter -- the name of the image file (str). If I am to run the script with the image `test.png`, I would enter the following command:
`python ocrtest.py test.png`
Note that by default, `ocrtest.py` activates CUDA. 
