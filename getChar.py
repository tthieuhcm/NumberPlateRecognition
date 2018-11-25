import cv2
import numpy as np
import logging
import matplotlib.pyplot as plt
from numpy.core.multiarray import ndarray

MAX_CHARS_IN_PLATE = 12


def isNearlyMaxLength(contour, image):
    # type: (ndarray, ndarray) -> int
    even = contour.flatten()[0::2]
    odd = contour.flatten()[1::2]
    if even.max() - even.min() > 0.4 * image.shape[0] or \
            odd.max() - odd.min() > 0.6 * image.shape[1]:
        return True
    else:
        return False


def isNearlyMinLength(contour, image):
    # type: (ndarray, ndarray) -> int
    even = contour.flatten()[0::2]
    odd = contour.flatten()[1::2]
    if (even.max() - even.min()) * (odd.max() - odd.min()) < 0.015 * image.shape[0] * image.shape[1]:
        return True
    else:
        return False


def getChars(img: ndarray = None):
    if img is None:
        logging.ERROR("No image has chosen !!!")

    # Convert to gray image
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Brightness balancing
    equal_histogram = cv2.equalizeHist(gray_img)

    # Noise removing
    noise_removal = cv2.bilateralFilter(equal_histogram, 5, 50, 50)

    # Edge Sharpening
    kernel3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    thre_mor = cv2.morphologyEx(noise_removal, cv2.MORPH_ERODE, kernel3)

    # Noise filtering
    # roi_blur = cv2.GaussianBlur(equal_histogram, (3, 3), 1)

    # Convert to Binary Image
    # ret, thre = cv2.threshold(thre_mor, 70, 255, cv2.THRESH_BINARY_INV)
    thre = cv2.adaptiveThreshold(thre_mor, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 41, 0)

    _, cont, hier = cv2.findContours(thre,
                                     cv2.RETR_CCOMP,
                                     cv2.CHAIN_APPROX_SIMPLE)

    # cv2.imshow("gray_img", gray_img)
    # cv2.imshow("equal_histogram", equal_histogram)
    # cv2.imshow("noise_removal", noise_removal)
    # cv2.imshow("thre_mor", thre_mor)
    # cv2.imshow("roi_blur", roi_blur)
    # cv2.imshow("Binary", thre)

    # cv2.drawContours(img, cont, -1, (0, 255, 0), 1)
    # cv2.imshow("Contour", img)
    # cv2.waitKey(0)

    areas_idx = {}
    areas = []
    for idx, cnt in enumerate(cont):
        if hier[0][idx][3] == -1 and \
                not isNearlyMaxLength(cnt, img) and \
                not isNearlyMinLength(cnt, img):
            # cv2.drawContours(img, cnt, -1, (0, 255, 0), 3)
            area = cv2.contourArea(cnt)
            if area in areas_idx:
                area += 1
            areas_idx[area] = idx
            areas.append(area)

    areas = sorted(areas, reverse=True)[0:MAX_CHARS_IN_PLATE]

    char_positions = list()
    for idx in areas:
        (x, y, w, h) = cv2.boundingRect(cont[areas_idx[idx]])
        # cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        char_positions.append(np.array([x, y, w, h]))

    char_positions = np.array(char_positions)
    char_positions = char_positions[char_positions[:, 0].argsort()[::1]]
    for position in char_positions:
        crop_img = img[position[1]:position[1] + position[3], position[0]:position[0] + position[2]]
        plt.imshow(cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB))
        plt.show()


if __name__ == '__main__':
    img = cv2.imread("./1/4BEE0140_51C-849.10_01022018011133_i3.jpg")
    # img = cv2.imread("./1/0EFE0140_61F4-0755_01022018124223_i3.jpg")
    # img = cv2.imread("./1/0EFE0140_61F4-9255_01022018202558_o3.jpg")
    # img = cv2.imread("./1/0EFE0140_61F4-9755_01022018165233_i3.jpg")
    # img = cv2.imread("./1/0EFE0140_67F4-9752_01022018161033_o3.jpg")
    # img = cv2.imread("./1/1EF00840_68T1-268.00_01022018190014_i3.jpg")
    # img_ = cv2.imread("./1/2AE90140_59X2-549_01022018134827_i3.jpg")
    # img = cv2.imread("./1/2AE90140_59X2-695.49_01022018143058_o3.jpg")
    # img = cv2.imread("./1/2C500040_61D1-535.82_01022018085147_i3.jpg")
    # img = cv2.imread("./1/2C500040_61D1-535.82_01022018164325_o3.jpg")
    # img = cv2.imread("./1/2C500040_61D1-535.82_01022018164331_o3.jpg")
    # img_ = cv2.imread("./1/2D4C0040_15B1-2328_01022018170910_o3.jpg")
    # img_ = cv2.imread("./1/2D4C0040_15B1-7971_01022018073014_i3.jpg")
    # img = cv2.imread("./1/2DF10840_51F-181.65_01022018062134_i3.jpg")
    # img = cv2.imread("./1/2DF10840_51F-181.65_01022018063831_i3.jpg")
    # img = cv2.imread("./1/2DF10840_51F-181.65_01022018090128_o3.jpg")
    # img = cv2.imread("./1/2DF10840_512TB-165_01022018063308_o3.jpg")
    # img = cv2.imread("./1/2DF10840_5040DG-851_01022018062100_i3.jpg")
    # img = cv2.imread("./1/3C690840_51C-045.12_01022018001929_i3.jpg")
    # img = cv2.imread("./1/3C690840_51C-045.12_01022018032955_o3.jpg")
    # img = cv2.imread("./1/3C690840_51C-045.12_01022018065439_i3.jpg")
    # img_ = cv2.imread("./1/3C690840_51C-045.12_01022018071031_o3.jpg")
    # img = cv2.imread("./1/4BEE0140_51C-849.10_01022018011133_i3.jpg")
    # img = cv2.imread("./1/4BEE0140_51C-849.10_01022018091304_i3.jpg")
    # img = cv2.imread("./1/4BEE0140_51C-849.10_01022018092938_o3.jpg")
    # img = cv2.imread("./1/4BEE0140_84R-10_01022018020813_o3.jpg")
    # img = cv2.imread("./1/04CF0140_30K1-8704_01022018143824_o3.jpg")
    # img = cv2.imread("./1/04CF0140_30K1-8704_01022018175227_i3.jpg")
    # img_ = cv2.imread("./1/4E830840_61LD-025.57_01022018084946_i3.jpg")
    # img = cv2.imread("./1/4E830840_61LD-025.57_01022018092032_o3.jpg")
    # img_ = cv2.imread("./1/4EE90140_61A1-003.34_01022018073856_o3.jpg")
    # img = cv2.imread("./1/4F980140_59Y1-823.05_01022018170942_o3.jpg")
    # img = cv2.imread("./1/4F980140_59Y1-824.05_01022018113937_i3.jpg")
    # img_ = cv2.imread("./1/4FD90840_51D-134.94_01022018080044_i3.jpg")
    # img = cv2.imread("./1/4FD90840_51D-134.94_01022018090139_o3.jpg")
    # img_ = cv2.imread("./1/4FD90840_51D-134.94_01022018143438_i3.jpg")
    # img = cv2.imread("./1/4FD90840_51D-134.94_01022018161339_o3.jpg")
    # img_ = cv2.imread("./1/06A01CBE_52-694_01022018082402_i3.jpg")
    # img_ = cv2.imread("./1/06A030BE_41L1-025.17_01022018181627_o3.jpg")
    # img_ = cv2.imread("./1/06A030BE_47L1-025.17_01022018060033_i3.jpg")
    # img_ = cv2.imread("./1/06A41BBE_45TC-201.54_01022018224734_i3.jpg")
    # img_ = cv2.imread("./1/06A43EBE_76M1-201.31_01022018070223_i3.jpg")
    # img_ = cv2.imread("./1/06A43EBE_76M1-201.39_01022018170427_o3.jpg")
    # img = cv2.imread("./1/06A45B0E_59B1-220.06_01022018073212_i3.jpg")
    # img = cv2.imread("./1/06A45B0E_59B1-220.06_01022018172320_o3.jpg")
    # img_ = cv2.imread("./1/06A61B10_63B1-261.94_01022018065143_i3.jpg")
    # img = cv2.imread("./1/06A61B10_261.94_01022018165530_o3.jpg")
    # img_ = cv2.imread("./1/06A1630E_42C-27_01022018174602_o3.jpg")
    # img_ = cv2.imread("./1/06A1630E_52K7-449_01022018080300_i3.jpg")
    # img_ = cv2.imread("./1/06A22110_67M4-3921_01022018232210_o3.jpg")
    # img_ = cv2.imread("./1/06A22110_69R6-3928_01022018074151_i3.jpg")
    # img_ = cv2.imread("./1/06A92410_60C2-130.66_01022018065242_i3.jpg")
    # img_ = cv2.imread("./1/06A92410_60C2-130.66_01022018161812_o3.jpg")
    # img = cv2.imread("./1/6AA90140_63Y-9029_01022018130713_i3.jpg")
    # img = cv2.imread("./1/6AA90140_63Y-9029_01022018161725_o3.jpg")
    # img_ = cv2.imread("./1/06AC2ABE_51C-587.84_01022018011711_i3.jpg")
    # img_ = cv2.imread("./1/06AC2ABE_51C-587.84_01022018035528_o3.jpg")
    # img = cv2.imread("./1/06AC670E_54U3-3687_01022018074230_i3.jpg")
    # img = cv2.imread("./1/06AC670E_54U3-3687_01022018175409_o3.jpg")
    # img = cv2.imread("./1/06AE2110_34B3-652.00_01022018081906_i3.jpg")
    # img = cv2.imread("./1/06AE2110_34B3-652.00_01022018095221_o3.jpg")
    # img = cv2.imread("./1/06B83A0E_67K1-397.59_01022018073732_i3.jpg")
    # img = cv2.imread("./1/06B83A0E_67K1-397.59_01022018112909_o3.jpg")
    # img = cv2.imread("./1/06B83A0E_67K1-397.59_01022018125303_i3.jpg")
    # img = cv2.imread("./1/06B83A0E_67K1-397.59_01022018164321_o3.jpg")
    # img_ = cv2.imread("./1/06B1650E_15TC-847.01_01022018022120_i3.jpg")
    # img_ = cv2.imread("./1/06B1650E_39F-39_01022018065525_i3.jpg")
    # img = cv2.imread("./1/06B1650E_51C-847.01_01022018032550_o3.jpg")
    # img = cv2.imread("./1/06B1650E_51C-847.01_01022018063100_i3.jpg")
    # img = cv2.imread("./1/06B1650E_51C-847.01_01022018064136_o3.jpg")
    # img = cv2.imread("./1/06B1650E_51C-877.04_01022018021351_o3.jpg")
    # img = cv2.imread("./1/06B1650E_51C-877.44_01022018011102_i3.jpg")
    # img_ = cv2.imread("./1/06BBEAC0_64G1-052.97_01022018054646_i3.jpg")
    # img_ = cv2.imread("./1/06BCF50F_67L1-833.34_01022018203646_i3.jpg")
    # img = cv2.imread("./1/06BF1E0E_77H1-305.97_01022018073711_i3.jpg")
    # img = cv2.imread("./1/06BF1E0E_77H1-305.97_01022018171710_o3.jpg")
    # img = cv2.imread("./1/06BF6C0E_01_01022018174327_i3.jpg")
    # img = cv2.imread("./1/06BF6C0E_61S4-0133_01022018022106_o3.jpg")
    # img = cv2.imread("./1/06BFE9C0_60B5-581.69_01022018054352_i3.jpg")
    # img = cv2.imread("./1/06BFE9C0_60B5-581.69_01022018054352_i3.jpg")
    # img_ = cv2.imread("./1/06C9540E_61C1-446.16_01022018071601_i3.jpg")
    # img_ = cv2.imread("./1/06C9540E_61C1-446.16_01022018130008_i3.jpg")
    # img_ = cv2.imread("./1/06C9540E_61C1-446.16_01022018162417_i3.jpg")
    # img = cv2.imread("./1/06CB6C0E_86B8-063.35_01022018085054_i3.jpg")
    # img = cv2.imread("./1/06CB6C0E_86B8-263.35_01022018165202_o3.jpg")
    # img_ = cv2.imread("./1/06CD670E_29C-528.06_01022018071923_i3.jpg")
    # img = cv2.imread("./1/06CD670E_528.06_01022018091203_o3.jpg")
    # img = cv2.imread("./1/06CFF60F_66P1-242.19_01022018074916_i3.jpg")
    # img = cv2.imread("./1/06CFF60F_66P7-242.19_01022018160807_o3.jpg")
    # img = cv2.imread("./1/06D7E8C0_49V5-6208_01022018174250_o3.jpg")
    # img = cv2.imread("./1/06D7E8C0_49V5-6208_01022018081341_i3.jpg")
    # img_ = cv2.imread("./1/06D603BE_51C-549.30_01022018091541_o3.jpg")
    # img_ = cv2.imread("./1/06D603BE_51C-549.30_01022018072035_i3.jpg")
    # img = cv2.imread("./1/06D72410_73H1-289.72_01022018192809_o3.jpg")
    # img = cv2.imread("./1/06D72410_73H1-489.72_01022018043903_i3.jpg")
    # img = cv2.imread("./1/06E06B0E_71H6-0129_01022018105506_o3.jpg")
    # img_ = cv2.imread("./1/06E06B0E_71H8-01_01022018060400_i3.jpg")
    # img = cv2.imread("./1/06E06B0E_71H8-0129_01022018112709_i3.jpg")
    # img = cv2.imread("./1/06E06B0E_0119_01022018145546_o3.jpg")
    # img_ = cv2.imread("./1/06E7E90D_24H-116.89_01022018174106_i3.jpg")
    # img_ = cv2.imread("./1/06E7E90D_24H-136.89_01022018075336_o3.jpg")
    getChars(img)
