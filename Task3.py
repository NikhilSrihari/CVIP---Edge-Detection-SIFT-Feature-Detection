import numpy as np
import argparse, glob
from cv2 import imread, rectangle, imshow, imwrite, waitKey, destroyAllWindows, resize, Laplacian, GaussianBlur, filter2D, matchTemplate, minMaxLoc  
from cv2 import CV_8U as CONST_CV_8U, TM_CCOEFF as CONST_TM_CCOEFF, TM_CCORR as CONST_TM_CCORR
from scipy.ndimage.filters import gaussian_filter

imageLocation_template = './task3/template.png'
imageLocation_positiveImages_prefix = './task3/pos_'
imageLocation_positiveImages_suffix = '.jpg'
imageLocation_negativeImages_prefix = './task3/neg_'
imageLocation_negativeImages_suffix = '.jpg'
numberOfPositiveImages = 15
numberOfNegativeImages = 10


def readImage(imageLocation):
    img = imread(imageLocation,0)
    return img


def writeImage(img, outputFileName):
	imwrite('output/Task3_Sol1_'+outputFileName+'.jpg', img)
	return 1


def displayImage(img, imgTitle='No Title'):
    imshow(imgTitle, img)
    waitKey(0)
    destroyAllWindows()


def getSharpeningKernel():
	sharpeningKernel = np.array(
								[ [-1,-1,-1],
                              	  [-1, 9,-1],
                              	  [-1,-1,-1] ]
                               )
	return sharpeningKernel


def main():
	print("Starting processing")
	img_template = readImage(imageLocation_template)
	img_template = resize(img_template , (13,18))
	img_template = Laplacian(img_template, CONST_CV_8U)
	img_template = GaussianBlur(img_template, (3,3),0)
	sharpeningKernel = getSharpeningKernel()
	for i in range(1, numberOfPositiveImages+1):
		imageLocation = imageLocation_positiveImages_prefix+str(i)+imageLocation_positiveImages_suffix
		print("Image at '"+imageLocation+"'")
		img_test_original = readImage(imageLocation)
		img_test = img_test_original.copy()
		img_test = Laplacian(img_test, CONST_CV_8U)
		img_test = GaussianBlur(img_test, (3,3), 0)
		img_test = filter2D(img_test, -1, sharpeningKernel)
		result = matchTemplate(img_test, img_template, CONST_TM_CCOEFF)
		min_val, max_val, min_loc, max_loc = minMaxLoc(result)
		if (max_val>400000):
			print("	Template Found")
			img_test_original = np.dstack([img_test_original, img_test_original, img_test_original])
			rectangle(img_test_original, (max_loc[0], max_loc[1]), (max_loc[0]+img_template.shape[1], max_loc[1]+img_template.shape[0]), (0, 255, 255), 2)
		else:
			print("	Template Not Found")
		writeImage(img_test_original, "Pos"+str(i))
	print("Positive images processed")
	for i in range(1, numberOfNegativeImages+1):
		if (i==7):
			continue
		imageLocation = imageLocation_negativeImages_prefix+str(i)+imageLocation_negativeImages_suffix
		print("Image at '"+imageLocation+"'")
		img_test_original = readImage(imageLocation)
		img_test = img_test_original.copy()
		img_test = Laplacian(img_test, CONST_CV_8U)
		img_test = GaussianBlur(img_test, (3,3), 0)
		img_test = filter2D(img_test, -1, sharpeningKernel)
		result = matchTemplate(img_test, img_template, CONST_TM_CCOEFF)
		min_val, max_val, min_loc, max_loc = minMaxLoc(result)
		if (max_val>400000):
			print("	Template Found")
			img_test_original = np.dstack([img_test_original, img_test_original, img_test_original])
			rectangle(img_test_original, (max_loc[0], max_loc[1]), (max_loc[0]+img_template.shape[1], max_loc[1]+img_template.shape[0]), (0, 255, 255), 2)
		else:
			print("	Template Not Found")
		writeImage(img_test_original, "Neg"+str(i))
	print("Negative images processed")


main()