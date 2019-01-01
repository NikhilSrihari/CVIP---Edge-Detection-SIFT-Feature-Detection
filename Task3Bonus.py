import numpy as np
import argparse, glob
from cv2 import imread, rectangle, imshow, imwrite, waitKey, destroyAllWindows, resize, Laplacian, GaussianBlur, filter2D, matchTemplate, minMaxLoc  
from cv2 import CV_8U as CONST_CV_8U, TM_CCOEFF as CONST_TM_CCOEFF, TM_CCORR as CONST_TM_CCORR
from scipy.ndimage.filters import gaussian_filter

imageLocation_template1 = './task3_bonus/template1.jpg'
imageLocation_positiveImages1_prefix = './task3_bonus/t1_'
imageLocation_positiveImages1_suffix = '.jpg'
imageLocation_template2 = './task3_bonus/template2.jpg'
imageLocation_positiveImages2_prefix = './task3_bonus/t2_'
imageLocation_positiveImages2_suffix = '.jpg'
imageLocation_template3 = './task3_bonus/template3.jpg'
imageLocation_positiveImages3_prefix = './task3_bonus/t3_'
imageLocation_positiveImages3_suffix = '.jpg'
imageLocation_negativeImages_prefix = './task3_bonus/neg_'
imageLocation_negativeImages_suffix = '.jpg'
template1_sizeArray = [30, 20, 25, 20, 30, 22]
template3_sizeArray = [30, 25, 30, 30, 30, 30]
numberOfPositive1Images = 6
numberOfPositive2Images = 6
numberOfPositive3Images = 6
numberOfNegativeImages = 12


def readImage(imageLocation):
    img = imread(imageLocation,0)
    return img


def writeImage(img, outputFileName):
	imwrite('output/Task3Bonus_'+outputFileName+'.jpg', img)
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
	print("Starting Template 1 processing")
	sharpeningKernel = getSharpeningKernel()
	for i in range(1, numberOfPositive1Images+1):
		img_template = readImage(imageLocation_template1)
		if (len(template1_sizeArray)>=numberOfPositive1Images):
			img_template = resize(img_template , (template1_sizeArray[i-1], template1_sizeArray[i-1]))
		else:
			img_template = resize(img_template , (20, 20))
		img_template = Laplacian(img_template, CONST_CV_8U)
		img_template = GaussianBlur(img_template, (3,3),0)
		imageLocation = imageLocation_positiveImages1_prefix+str(i)+imageLocation_positiveImages1_suffix
		print("Image at '"+imageLocation+"'")
		img_test_original = readImage(imageLocation)
		img_test = img_test_original.copy()
		img_test = Laplacian(img_test, CONST_CV_8U)
		img_test = GaussianBlur(img_test, (3,3), 0)
		img_test = filter2D(img_test, -1, sharpeningKernel)
		result = matchTemplate(img_test, img_template, CONST_TM_CCOEFF)
		min_val, max_val, min_loc, max_loc = minMaxLoc(result)
		print(max_val)
		if (max_val>400000):
			print("	Template Found")
			img_test_original = np.dstack([img_test_original, img_test_original, img_test_original])
			rectangle(img_test_original, (max_loc[0], max_loc[1]), (max_loc[0]+img_template.shape[1], max_loc[1]+img_template.shape[0]), (255, 0, 0), 2)
		else:
			print("	Template Not Found")
		writeImage(img_test_original, "t1_"+str(i))
	print("Template 1 positive images processed")
	print("Starting Template 2 processing")
	sharpeningKernel = getSharpeningKernel()
	for i in range(1, numberOfPositive2Images+1):
		img_template = readImage(imageLocation_template2)
		img_template = resize(img_template , (30,30))
		img_template = Laplacian(img_template, CONST_CV_8U)
		img_template = GaussianBlur(img_template, (3,3),0)
		imageLocation = imageLocation_positiveImages2_prefix+str(i)+imageLocation_positiveImages2_suffix
		print("Image at '"+imageLocation+"'")
		img_test_original = readImage(imageLocation)
		img_test = img_test_original.copy()
		img_test = Laplacian(img_test, CONST_CV_8U)
		img_test = GaussianBlur(img_test, (3,3), 0)
		img_test = filter2D(img_test, -1, sharpeningKernel)
		result = matchTemplate(img_test, img_template, CONST_TM_CCOEFF)
		min_val, max_val, min_loc, max_loc = minMaxLoc(result)
		print(max_val)
		if (max_val>400000):
			print("	Template Found")
			img_test_original = np.dstack([img_test_original, img_test_original, img_test_original])
			rectangle(img_test_original, (max_loc[0], max_loc[1]), (max_loc[0]+img_template.shape[1], max_loc[1]+img_template.shape[0]), (0, 255, 0), 2)
		else:
			print("	Template Not Found")
		writeImage(img_test_original, "t2_"+str(i))
	print("Template 2 positive images processed")
	print("Starting Template 3 processing")
	sharpeningKernel = getSharpeningKernel()
	for i in range(1, numberOfPositive3Images+1):
		img_template = readImage(imageLocation_template3)
		if (len(template3_sizeArray)>=numberOfPositive3Images):
			img_template = resize(img_template , (template3_sizeArray[i-1], template3_sizeArray[i-1]))
		else:
			img_template = resize(img_template , (30, 30))
		img_template = Laplacian(img_template, CONST_CV_8U)
		img_template = GaussianBlur(img_template, (3,3),0)
		imageLocation = imageLocation_positiveImages3_prefix+str(i)+imageLocation_positiveImages3_suffix
		print("Image at '"+imageLocation+"'")
		img_test_original = readImage(imageLocation)
		img_test = img_test_original.copy()
		img_test = Laplacian(img_test, CONST_CV_8U)
		img_test = GaussianBlur(img_test, (3,3), 0)
		img_test = filter2D(img_test, -1, sharpeningKernel)
		result = matchTemplate(img_test, img_template, CONST_TM_CCOEFF)
		min_val, max_val, min_loc, max_loc = minMaxLoc(result)
		print(max_val)
		if (max_val>400000):
			print("	Template Found")
			img_test_original = np.dstack([img_test_original, img_test_original, img_test_original])
			rectangle(img_test_original, (max_loc[0], max_loc[1]), (max_loc[0]+img_template.shape[1], max_loc[1]+img_template.shape[0]), (0, 0, 255), 2)
		else:
			print("	Template Not Found")
		writeImage(img_test_original, "t3_"+str(i))
	print("Template 3 positive images processed")
	
	print("Starting Negative Image processing against all templates")
	for i in range(1, numberOfNegativeImages+1):
		img_template1 = readImage(imageLocation_template1)
		img_template1 = resize(img_template1 , (20,20))
		img_template1 = Laplacian(img_template1, CONST_CV_8U)
		img_template1 = GaussianBlur(img_template1, (3,3),0)
		img_template2 = readImage(imageLocation_template2)
		img_template2 = resize(img_template2 , (20,20))
		img_template2 = Laplacian(img_template2, CONST_CV_8U)
		img_template2 = GaussianBlur(img_template2, (3,3),0)
		img_template3 = readImage(imageLocation_template3)
		img_template3 = resize(img_template3 , (20,20))
		img_template3 = Laplacian(img_template3, CONST_CV_8U)
		img_template3 = GaussianBlur(img_template3, (3,3),0)
		if (i==7):
			continue
		imageLocation = imageLocation_negativeImages_prefix+str(i)+imageLocation_negativeImages_suffix
		print("Image at '"+imageLocation+"'")
		img_test_original = readImage(imageLocation)
		img_test = img_test_original.copy()
		img_test = Laplacian(img_test, CONST_CV_8U)
		img_test = GaussianBlur(img_test, (3,3), 0)
		img_test = filter2D(img_test, -1, sharpeningKernel)
		result1 = matchTemplate(img_test, img_template1, CONST_TM_CCOEFF)
		result2 = matchTemplate(img_test, img_template2, CONST_TM_CCOEFF)
		result3 = matchTemplate(img_test, img_template3, CONST_TM_CCOEFF)
		min_val1, max_val1, min_loc1, max_loc1 = minMaxLoc(result1)
		min_val2, max_val2, min_loc2, max_loc2 = minMaxLoc(result2)
		min_val3, max_val3, min_loc3, max_loc3 = minMaxLoc(result3)
		img_test_original = np.dstack([img_test_original, img_test_original, img_test_original])
		print(max_val1)
		print(max_val2)
		print(max_val3)
		if (max_val1>400000):
			print("	Template1 Found")
			rectangle(img_test_original, (max_loc1[0], max_loc1[1]), (max_loc1[0]+img_template1.shape[1], max_loc1[1]+img_template1.shape[0]), (255, 0, 0), 2)
		else:
			print("	Template1 Not Found")
		if (max_val2>400000):
			print("	Template2 Found")
			rectangle(img_test_original, (max_loc2[0], max_loc2[1]), (max_loc2[0]+img_template2.shape[1], max_loc2[1]+img_template2.shape[0]), (0, 255, 0), 2)
		else:
			print("	Template3 Not Found")
		if (max_val3>400000):
			print("	Template3 Found")
			rectangle(img_test_original, (max_loc3[0], max_loc3[1]), (max_loc3[0]+img_template3.shape[1], max_loc3[1]+img_template3.shape[0]), (0, 0, 255), 2)
		else:
			print("	Template3 Not Found")
		writeImage(img_test_original, "Neg"+str(i))
	print("Negative images processed")


main()