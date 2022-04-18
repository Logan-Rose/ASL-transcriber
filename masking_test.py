import cv2 as cv
import numpy as np
import os


def open_image(path):
    img = cv.imread(path)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    return img, gray


def open_operation(ex_img):
    ex_img = cv.erode(ex_img, (11, 11))
    ex_img = cv.dilate(ex_img, (11, 11))
    return ex_img


def prepare_image(grayImg):
    img1 = cv.medianBlur(grayImg, 13)
    ex_img = open_operation(img1)
    ex_img = open_operation(ex_img)
    ex_img = open_operation(ex_img)
    return ex_img

def contrast(img):
    _, mask = cv.threshold(img, thresh=50, maxval=255, type=cv.THRESH_OTSU)
    ex_img = prepare_image(mask)
    res = cv.bitwise_and(img, ex_img)
    cv.imshow("masked", res)
    cv.imshow("mask", mask)
    cv.imshow("unmasked", img)
    cv.waitKey(0)
    return res


def main():

    # initializations

    reg, grey = open_image("./frames/0.png")
    analysis = contrast(grey)

    cv.imshow("test", analysis)
    cv.waitKey(0)

    cv.destroyAllWindows()

main()