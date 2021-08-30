def parse_xml(xml):
    """
    parse xml text to be accessed through xpath
    :param xml: xml text
    :type xml: str
    :return: parsel.selector.Selector
    :rtype: object
    """
    # setup xml parser
    parsel.Selector.__str__ = parsel.Selector.extract
    parsel.Selector.__repr__ = parsel.Selector.__str__
    parsel.SelectorList.__repr__ = lambda x: '[{}]'.format(
        '\n '.join("({}) {!r}".format(i, repr(s))
                   for i, s in enumerate(x, start=1))
    ).replace(r'\n', '\n')

    doc = parsel.Selector(text=xml)
    return doc

def tesseract_read(file: str):
    img = cv2.imread(file)

    # img_edges = cv2.Canny(img, 100, 100, apertureSize=3)
    # lines = cv2.HoughLinesP(img_edges, 1, math.pi / 180.0, 100, minLineLength=100, maxLineGap=5)

    # angles = []
    # if lines is not None and lines.shape[0] > 0:
    #     for [[x1, y1, x2, y2]] in lines:
    #         cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 3)
    #         angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
    #         angles.append(angle)

    #     cv2.imshow("Detected lines", img) 

    #     median_angle = np.median(angles)
    #     img = ndimage.rotate(img, median_angle)
    # else:
    #     print("no lines detected")

    hocr = pytesseract.image_to_pdf_or_hocr(img, extension='hocr')
    xml = hocr.decode('utf-8')
    doc = parse_xml(xml)
    tsa_output = []

    # get text
    for tag in doc.xpath('/html/body/div/div/p/span/span'):
        tsa_output.append(str(tag.xpath('text()')[0]))

    tsa_output = " ".join(tsa_output).lower()  # lowercase the output to be better compared with easyocr
    tsa_output = tsa_output.replace('-', ' ')  # replace '-' to be better compared with easyocr
    
    return tsa_output

def easyocr_read(file: str, reader):
    results = reader.readtext(file)
    results = sorted(results, key=lambda x: x[0][0])  # sort text from left to right
    text_results = [x[-2] for x in results]  # get text
    easy_output = " ".join(text_results)  # join together
    easy_output = easy_output.strip()  # clean up spaces
    # easy_output = reader.sub('\s{2,}', ' ', easy_output)  # clean up spaces
    
    return easy_output



if __name__ == '__main__':
    from random_words import RandomWords
    import cv2
    import time
    import pytesseract
    import easyocr
    import parsel
    import numpy as np
    import pandas as pd
    import sys
    
    reader = easyocr.Reader(['en'], gpu = bool(int(sys.argv[2])))

    try:
        test_size = int(sys.argv[1])
    except:
        test_size = 1000

    file = str(sys.argv[3])
    answer_list = []
    tsa_list = []
    easy_list = []
    tsa_time_used = 0
    easy_time_used = 0
    rw = RandomWords()
    for i in range(test_size):
        print(i)

        # create_img(file, text)  # create the image with text
        
        # get result from tesseract
        tsa_start_time = time.time()
        tsa_result = tesseract_read(file)
        tsa_end_time = time.time()
        tsa_time_used = tsa_time_used + (tsa_end_time-tsa_start_time)
        
        # get result from easyocr
        easy_start_time = time.time()
        easy_result = easyocr_read(file, reader)
        easy_end_time = time.time()
        easy_time_used = easy_time_used + (easy_end_time-easy_start_time)
        
        # append result
        if i == test_size - 1:
            tsa_list.append(tsa_result)
            easy_list.append(easy_result)

    # convert to array
    tsa_array = np.array(tsa_list)
    easy_array = np.array(easy_list)

    # print out error rates
    print(f"Tesseract on {test_size} samples: averages {tsa_time_used / test_size} seconds")
    print("Final Tesseract output: ")
    print(tsa_array)
    print(f"EasyOCR on {test_size} samples: averages {easy_time_used / test_size} seconds")
    print("Final Easy output: ")
    print(easy_array)
