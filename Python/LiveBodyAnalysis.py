import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys
import argparse
import scipy.ndimage as sp
import SegmentAnalysis as SA

# Set up cmd arguments for the user
parser = argparse.ArgumentParser(description = 'Enter folder path, base file name, number of images, and file type')
parser.add_argument('path', type = str, help = 'Relative folder path (ex. WT225/)')
parser.add_argument('base_name', type = str, help = 'Base name of each image (ex. WT225_Snapshot)')
parser.add_argument('first_num', type = int, help = 'First image number')
parser.add_argument('last_num', type = int, help = 'Last image number')
parser.add_argument('file_type', type = str, help = 'File type of images (ex. png or jpg)')
args = parser.parse_args()

for num in range(args.first_num, args.last_num + 1):
    #####################################################################################
    # Step 1: read image
    #####################################################################################

    # Process a single image: read in based on the user input
    file_name = args.path + args.base_name + str(num) + '.' + args.file_type
    img = cv2.imread(file_name, cv2.IMREAD_GRAYSCALE)

    # If image cannot be processed, exit program
    if img is None:
        sys.exit("Could not read the image.")

    # Increase brightness of worms for next step (without this, our worm pictures just show
    # up as black rectangles)
    temp = cv2.imread(file_name)
    hsv = cv2.cvtColor(temp, cv2.COLOR_BGR2HSV)
    hsv[:,:,2] += 50
    see_img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    # Prompt user to select the background, stored in bkCoords of type tuple. cv2.selectROI
    # takes in a window name and the desired image and will return (left_col, top_row,
    # right_col, bottom_row).
    bkCoords = cv2.selectROI("Please select a background region. Then press enter or space.", see_img)
    bk = img[int(bkCoords[1]):int(bkCoords[1] + bkCoords[3]), int(bkCoords[0]):int(bkCoords[0] + bkCoords[2])]
    # cv2.imshow("Selected background. Press 's' when finished.", bk)
    # cv2.waitKey(0)

    #####################################################################################
    # Step 2: threshold image
    #####################################################################################

    # Otsu's thresholding to get the image to be binary black and white.
    #   Otsu's thresholding automatically calculates a threshold value
    #   from image histogram for a bimodal image. This works for our images
    #   because we have the worm (1) and the background (0).
    # cv2 function "threshold" uses Otsu's thresholding. If Otsu's thresholding
    # is NOT used, ret is the threshold value you entered (which would be 0 here).
    # Parameters: cv2.threshold(grayscale image, threshold, value given to pixels above
    #   threshold, threshold type)
    ret,th = cv2.threshold(img,0,1,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    # ret is type float representing the chosen threshold
    # th is a numpy array of 0s and 1s of your original image

    #########################################
    # FOLLOWING CODE FROM WEBSITE
    # https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_thresholding/py_thresholding.html
    # global thresholding
    ret1,th1 = cv2.threshold(img,50,255,cv2.THRESH_BINARY)

    # Otsu's thresholding
    ret2,th2 = cv2.threshold(img,0,1,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    # Otsu's thresholding after Gaussian filtering
    blur = cv2.GaussianBlur(img,(5,5),0)
    ret3,th3 = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    images = [img, 0, th1,
              img, 0, th2,
              blur, 0, th3]
    titles = ['Original Noisy Image','Histogram','Global Thresholding (v=50)',
              'Original Noisy Image','Histogram',"Otsu's Thresholding",
              'Gaussian filtered Image','Histogram',"Otsu's Thresholding"]

    # for i in range(3):
    #     plt.subplot(3,3,i*3+1),plt.imshow(images[i*3],'gray')
    #     plt.title(titles[i*3]), plt.xticks([]), plt.yticks([])
    #     plt.subplot(3,3,i*3+2),plt.hist(images[i*3].ravel(),256)
    #     plt.title(titles[i*3+1]), plt.xticks([]), plt.yticks([])
    #     plt.subplot(3,3,i*3+3),plt.imshow(images[i*3+2],'gray')
    #     plt.title(titles[i*3+2]), plt.xticks([]), plt.yticks([])
    # plt.show()
    #########################################

    # Take Otsu's thresholded image, th2, and do dilating/morphological opening.
    # No function like matlab's imfill exists in cv2.
    # Kernel size determines how much/little the image is dilated/opened.
    kernel_2 = np.ones((2,2), np.uint8)
    kernel_3 = np.ones((3,3), np.uint8)
    kernel_5 = np.ones((5,5), np.uint8)
    kernel_6 = np.ones((6,6), np.uint8)
    kernel_10 = np.ones((10,10), np.uint8)
    kernel_15 = np.ones((15,15), np.uint8)
    ImDia = cv2.dilate(th2, kernel_15, iterations = 1)
    ImOpen = cv2.morphologyEx(ImDia, cv2.MORPH_OPEN, kernel_15)

    # optional: plot the results
    # plt.subplot(3,1,1)
    # plt.imshow(th2)
    # plt.title('Thresholded Image')
    # plt.subplot(3,1,2)
    # plt.imshow(ImDia)
    # plt.title('Dilated Image (kernel size = 15)')
    # plt.subplot(3,1,3)
    # plt.imshow(ImOpen)
    # plt.title('Image after Morphological Opening (kernel size = 15)')
    # plt.subplots_adjust(wspace = 0.6, hspace = 0.6)
    # plt.show()

    #####################################################################################
    # Step 3: isolate worm
    #####################################################################################

    # Identify connected regions and select the largest, which we assume to be the worm.
    # labeled_array is the matrix of the image, containing integers corresponding to region
    #    0, region 1, region 2, etc.
    # num_regions is the number of regions found (excluding 0)
    labeled_array, num_regions = sp.label(ImOpen)
    max_size = 0
    max_region = 0
    # Iterate through the regions identified by sp.label, excluding 0, the background
    for x in range(1, num_regions + 1):
        # Get the size of each region (count) and update the maximums
        count = np.count_nonzero(labeled_array == x)
        if (count > max_size):
            max_size = count
            max_region = x

    # Isolate the desired region by dividing each pixel by the number of the largest
    # region. If a pixel belongs to the largest region, this value will be one.
    # Generate a matrix of True and False, and turn it into 0s and 1s using .astype(int)
    justWorm = (np.divide(labeled_array, max_region) == 1).astype(int)

    # Multiply your binary mask by the original image to get just the worm body
    ImSegGray = np.multiply(justWorm, img)

    # Plot resulting image
    # worm = np.divide(ImSegGray, 255)
    # plt.subplot(2,1,1)
    # plt.imshow(img)
    # plt.title('Original Image')
    # plt.subplot(2,1,2)
    # plt.imshow(worm)
    # plt.title('Worm Body Only')
    # plt.subplots_adjust(wspace = 0.6, hspace = 0.6)
    # plt.show()

    #####################################################################################
    # Step 4: feature detection
    #####################################################################################

    # Is the image cut off? Is there a nonzero number in any of the outer rows/columns?
    cut_off = False
    row1 = ImSegGray[0]
    row2 = ImSegGray[ImSegGray.shape[0] - 1]
    for row in ImSegGray:
        if row[0] >= 1:
            cut_off = True
        if row[ImSegGray.shape[1] - 1] >= 1:
            cut_off = True
    for position in row1:
        if position >= 1:
            cut_off = True
    for position in row2:
        if position >= 1:
            cut_off = True

    # Estimate worm length: np.nonzero() identifies the coordinates of all nonzero
    # elements
    too_short = False
    [row,col] = np.nonzero(ImSegGray)
    est_len = max(col) - min(col)
    # worm should take up more than half of the full image length
    if est_len < img.shape[1]/2:
        too_short = True

    # Estimate the mean intensity of the worm and background
    bkMeanInt = np.mean(bk)
    wormMeanInt = np.mean(ImSegGray[np.nonzero(ImSegGray)])

    # H/T differentiation then re-orientation
    head_first = False
    values = [[],[],[],[],[]]
    for i in range(10):
        # divide the worm body into 10 equal segments according to estimated length, est_len
        SecImage = ImSegGray[min(row):max(row),(min(col)+(i)*est_len//10):(min(col)+(i+1)*est_len//10)]
        [MeanInt, avgWidth, stdInt, perct95, medianInt] = SA.stats(SecImage)
        values[0].append(MeanInt)
        values[1].append(avgWidth)
        values[2].append(stdInt)
        values[3].append(perct95)
        values[4].append(medianInt)
    # Ratio of mean intensity of segment 2 vs segment 9 gives us a good indication
    # of where the nerve ring (high intensity, located toward the head of the animal)
    # is fixed in the image.
    seg2int = values[0][1]
    seg9int = values[0][8]
    if seg9int/seg2int > 1.3:
        head_first = True

