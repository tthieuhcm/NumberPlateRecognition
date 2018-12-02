import numpy as np
import cv2
import imutils
import copy
import os
from random import shuffle
from OCR import readFolderAndClassify
from getChar import getChars, getLabel
from utils import list_files


def sharpening(img, level=5):
    gaussian_blur = cv2.GaussianBlur(img, (level, level), 50)
    sharpened = cv2.addWeighted(img, 1.5, gaussian_blur, -0.6, 0)
    # cv2.imshow('Sharpened iMG', sharpened)
    return sharpened


def preprocessing(img):
    # resized_img = imutils.resize(img, 640)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img = clahe.apply(gray_img)
    # cv2.equalizeHist(normalized_img, normalized_img)
    # cv2.imshow('normalize_img', normalized_img)
    return img


def closing(img, kernel_size=3, iterations=1):
    kernel = np.ones((kernel_size, kernel_size))
    dilation = cv2.dilate(img, kernel, iterations=iterations)
    erosion = cv2.erode(dilation, kernel, iterations=iterations)
    # cv2.imshow('dialated ', dilation)
    # cv2.imshow('erosed', erosion)
    return erosion


def opening(img, kernel_size=3, iterations=1):
    kernel = np.ones((kernel_size, kernel_size))
    erosion = cv2.erode(img, kernel, iterations=iterations)
    dilation = cv2.dilate(erosion, kernel, iterations=iterations)

    # cv2.imshow('dialated ', dilation)
    # cv2.imshow('erosed', erosion)
    return dilation


def noise_reducting(img, brush_size=15, intensity=30):
    noise_reducted = cv2.bilateralFilter(img, -1, intensity, brush_size)

    # cv2.imshow("Bilateral Filter ", noise_reducted)
    return noise_reducted


def plate_detection(in_img, img_name):
    in_img = imutils.resize(in_img, 800)
    crop_top = in_img.shape[0] // 8
    crop_bot = in_img.shape[0] - crop_top
    crop_left = in_img.shape[1] // 8
    crop_right = in_img.shape[1] - crop_left
    in_img = in_img[crop_top:crop_bot, crop_left:crop_right]
    # cv2.imshow("based img", in_img)

    img = preprocessing(in_img)

    # cv2.imshow('equalized', img)
    img = sharpening(img, level=45)

    img = noise_reducting(img, brush_size=30, intensity=70)

    # img = opening(img, iterations=2)
    # Find Edges of the grayscale image

    img = cv2.Canny(img, 110, 200, L2gradient=True)
    # cv2.imshow("Canny Edges", img)

    # # Increase edge width to fill hole
    img = closing(img, iterations=1)
    # Find contours based on Edges
    (new, cnts, _) = cv2.findContours(img.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=False)
    NumberPlateCnt = None
    all_contour = []
    # loop over our contours to find the best possible approximate contour of number plate
    count = 0
    for index, c in enumerate(cnts):
        peri = cv2.arcLength(c, True)

        approx = cv2.approxPolyDP(c, 0.07 * peri, True)

        # print(approx)
        # rectan = cv2.boundingRect(approx)
        # convex = cv2.convexHull(approx)
        # all_contour.append(approx)
        x, y, w, h = cv2.boundingRect(approx)
        # ratio = w/h

        if ((w > 140 and h < 70) or (h >= 70 and w > 90)) and len(approx) == 4 and cv2.isContourConvex(
                approx):  # Select the contour with 4 corners
            NumberPlateCnt = approx  # This is our approx Number Plate Contour
            cropped_plate = in_img[y:y + h, x:x + w]
            # print(cur_path + 'cropped2/' + img_name)
            cv2.imwrite(cur_path + 'cropped/' + img_name, cropped_plate)
            break

    # if NumberPlateCnt is not None :
    #     # cv2.imshow('crop', cv2.findNonZero(mask))
    #     if h >= 75 or w > 140 :
    #
    #     # cv2.drawContours(in_img, [NumberPlateCnt], -1, (0, 255, 0), 3)
    #     # cv2.imshow("Final Image With Number Plate Detected ", in_img)

    # cv2.drawContours(in_img, all_contour, -1, (0, 0, 255), 3)
    # cv2.imshow("All contour ", in_img)
    # cv2.waitKey(0) #Wait for user input before closing the images displayed


def batch_plate_detect(img_folder, output_folder):
    for i in img_file:
        path = cur_path + 'whole_plate/' + i
        # path = 'whole_plate/060D1C10_51D-025.49_01022018114253_o2.jpg'
        print(path)
        img = cv2.imread(path)
        plate_detection(img, i)
        # cv2.waitKey(0)


if __name__ == '__main__':
    cur_path = './FullPics/'
    img_file = [f for f in os.listdir(cur_path) if os.path.isfile(cur_path + f)]
    # img_file = [f for f in os.listdir(cur_path + 'Bike_back/') if os.path.isfile(cur_path + 'Bike_back/' + f)]
    # shuffle(img_file)
    index = 0
    for i in img_file[:1000]:
        path = cur_path + i
        # path = 'whole_plate/060D1C10_51D-025.49_01022018114253_o2.jpg'
        # print(path)
        img = cv2.imread(path)
        plate_detection(img, i)
        # preprocessing(img)
        # cv2.waitKey(0)
        index += 1
        # if index == 1000:
        #     break

    path_name = cur_path + 'cropped'
    total = 0
    hit = 0
    for f in list_files(path_name):
        total += 1
        path = getChars(f, path_name,
                        add_contour_dots_to_img=False,
                        show_more_info_pics=False,
                        draw_contour_rectangle_to_img=False,
                        show_image=False,
                        crop_threshold_img=False,
                        padding_cropped_img=False)
        if path != '':
            result = readFolderAndClassify(path)
            if result == getLabel(f):
                hit += 1
    print(hit / total)
