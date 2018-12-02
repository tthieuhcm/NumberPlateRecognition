import errno
import cv2
import numpy as np
import logging
import matplotlib.pyplot as plt
from numpy.core.multiarray import ndarray
import os
import re
from utils import list_files
from OCR import readFolderAndClassify

MAX_CHARS_IN_PLATE = 12
FIXED_OUTPUT_SIZE = 20


def isNearlyMaxLength(contour, image):
    # type: (ndarray, ndarray) -> int
    even = contour.flatten()[0::2]  # X-axis
    odd = contour.flatten()[1::2]  # Y-axis
    adaptThreshold = 0.57
    if 2 * image.shape[0] < image.shape[1]:
        adaptThreshold = 0.7
    if even.max() - even.min() > 0.4 * image.shape[1] or \
            odd.max() - odd.min() > adaptThreshold * image.shape[0]:
        return True
    else:
        return False


def isNearlyMinLength(contour, image):
    # type: (ndarray, ndarray) -> bool
    even = contour.flatten()[0::2]  # X-axis
    odd = contour.flatten()[1::2]  # Y-axis
    adaptThreshold = 0.068
    if 2 * image.shape[0] < image.shape[1]:
        adaptThreshold = 0.03
    if (even.max() - even.min()) * (odd.max() - odd.min()) < 0.01 * image.shape[0] * image.shape[1] or \
            even.max() - even.min() < adaptThreshold * image.shape[1] or \
            odd.max() - odd.min() < 0.1 * image.shape[0]:
        return True
    else:
        return False


def tooThin(contour, image):
    # type: (ndarray, ndarray) -> bool
    even = contour.flatten()[0::2]  # X-axis
    odd = contour.flatten()[1::2]  # Y-axis
    adaptThreshold = 5
    if 2 * image.shape[0] < image.shape[1]:
        adaptThreshold = 20
    if (even.max() - even.min()) > adaptThreshold * (odd.max() - odd.min()) or \
            (odd.max() - odd.min()) > adaptThreshold * (even.max() - even.min()):
        return True
    else:
        return False


def isLyingRectangle(contour):
    # type: (ndarray) -> bool
    even = contour.flatten()[0::2]  # X-axis
    odd = contour.flatten()[1::2]  # Y-axis
    if (even.max() - even.min()) > (odd.max() - odd.min()):
        return True
    else:
        return False


def getLabel(file_path=''):
    # type: (str) -> str
    try:
        start = file_path.find('_') + 1
        end = file_path.find('_', start)
        found = file_path[start:end]
        found = re.sub('[-.]', '', found)
    except AttributeError:
        found = ''  # apply your error handling
    return found


def getChars(img_path='',
             pth_name='',
             add_contour_dots_to_img=False,
             show_more_info_pics=False,
             draw_contour_rectangle_to_img=False,
             show_image=False,
             crop_threshold_img=False,
             padding_cropped_img=True,
             **kwargs):
    # type: (str, str, bool, bool, bool, bool, bool, bool, ...) -> str

    if img_path == '':
        logging.ERROR("No image has chosen !!!")

    image = cv2.imread(img_path)

    if image is None:
        print(img_path)
        logging.ERROR("Can't load image !!!")
        return ''

    # Convert to gray image
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Brightness balancing
    # equal_histogram = cv2.equalizeHist(gray_img)

    # Noise removing
    # noise_removal = cv2.medianBlur(gray_img, 3)

    # Edge Sharpening
    # kernel3 = np.ones((3, 3), np.uint8)
    # edge_sharpening = cv2.dilate(noise_removal, kernel3, iterations = 1)

    # Convert to Binary Image
    # thre = cv2.adaptiveThreshold(noise_removal, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 41, 0)
    noise_removal = cv2.GaussianBlur(gray_img, (5, 5), 0)
    _, thre = cv2.threshold(noise_removal, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    _, cont, hier = cv2.findContours(thre,
                                     cv2.RETR_CCOMP,
                                     cv2.CHAIN_APPROX_SIMPLE)
    if show_more_info_pics:
        cv2.imshow("Gray_img", gray_img)
        # cv2.imshow("Equal_histogram", thre_equal_histogram)
        cv2.imshow("Noise_removal", noise_removal)
        # cv2.imshow("Edge_sharpening", thre_edge_sharpening)
        cv2.imshow("OtsuThreshold", thre)

        if add_contour_dots_to_img:
            cv2.drawContours(image, cont, -1, (0, 255, 0), 1)
            cv2.imshow("Contour", image)
        cv2.waitKey(0)

    areas_idx = {}
    areas = []
    for idx, cnt in enumerate(cont):
        # cv2.drawContours(image, cnt, -1, (0, 255, 0), 1)
        # cv2.imshow("Contour", image)
        # cv2.waitKey(0)

        if hier[0][idx][3] == -1 and \
                not isNearlyMaxLength(cnt, image) and \
                not isNearlyMinLength(cnt, image) and \
                not tooThin(cnt, image) and \
                not isLyingRectangle(cnt):

            area = cv2.contourArea(cnt)
            if area in areas_idx:
                area += 1
            areas_idx[area] = idx
            areas.append(area)

    areas = sorted(areas, reverse=True)[0:MAX_CHARS_IN_PLATE]

    char_positions = list()
    for idx in areas:
        (x, y, w, h) = cv2.boundingRect(cont[areas_idx[idx]])
        if draw_contour_rectangle_to_img:
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        char_positions.append(np.array([x, y, w, h]))

    if show_image:
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.show()

    if char_positions.__len__() == 0:
        logging.info(img_path + " has no contour. Need to check again.")
        return ''

    # Sort the position of characters
    char_positions = np.array(char_positions)
    char_positions = char_positions[(char_positions[:, 0] + char_positions[:, 1] * 3).argsort()[::1]]

    for idx, position in enumerate(char_positions):
        if crop_threshold_img:
            crop_img = thre[position[1]:position[1] + position[3], position[0]:position[0] + position[2]]
        else:
            crop_img = image[position[1]:position[1] + position[3], position[0]:position[0] + position[2]]

        if padding_cropped_img:
            max_size = max(crop_img.shape[0], crop_img.shape[1])
            delta_w = max_size - crop_img.shape[1]
            delta_h = max_size - crop_img.shape[0]
            top, bottom = delta_h // 2, delta_h - (delta_h // 2)
            left, right = delta_w // 2, delta_w - (delta_w // 2)
            WHITE = [255, 255, 255]
            resized_image = cv2.copyMakeBorder(crop_img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=WHITE)
        else:
            resized_image = cv2.resize(crop_img, (FIXED_OUTPUT_SIZE, FIXED_OUTPUT_SIZE))

        # resized_image = cv2.bitwise_not(resized_image)
        # folder_name = getLabel(img_path)
        # if folder_name == '':
        #     return
        try:
            # os.makedirs(pth_name + folder_name)
            os.makedirs(img_path[:-4])

        except OSError as e:
            if e.errno != errno.EEXIST:
                raise
        # cv2.imwrite(pth_name + folder_name + '/' + str(idx) + '.jpg', crop_img)
        cv2.imwrite(img_path[:-4] + '/' + str(idx) + '.jpg', resized_image)
    return img_path[:-4]


if __name__ == '__main__':
    path_name = "./numberplates"
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
            # print("result: ", result)
            # print("target: ", getLabel(f))

            if result == getLabel(f):
                hit += 1
    print(hit/total)
