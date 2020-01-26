# By submitting this assignment, I agree to the following:
#  “Aggies do not lie, cheat, or steal, or tolerate those who do”
#  “I have not given or received any unauthorized aid on this assignment”
#
# Name: 		Shion Ito, Anh Hoang, Anh Nguyen, Huong Vo
# Assignment:	TAMUHACK <3
# Date:		January 26, 2018

from __future__ import print_function
import cv2 as cv
import argparse
import pyautogui

detector_params = cv.SimpleBlobDetector_Params()
detector_params.filterByArea = True
detector_params.maxArea = 1500
detector = cv.SimpleBlobDetector_create(detector_params)

parser = argparse.ArgumentParser(description='Code for Cascade Classifier tutorial.')
parser.add_argument('--face_cascade', help='Path to face cascade.', default='data/haarcascade_frontalface_alt.xml')
parser.add_argument('--eyes_cascade', help='Path to eyes cascade.', default='data/haarcascade_eye_tree_eyeglasses.xml')
parser.add_argument('--camera', help='Camera divide number.', type=int, default=0)
args = parser.parse_args()
face_cascade_name = args.face_cascade
eyes_cascade_name = args.eyes_cascade
face_cascade = cv.CascadeClassifier('data/haarcascade_frontalface_alt.xml')
eyes_cascade = cv.CascadeClassifier('data/haarcascade_eye_tree_eyeglasses.xml')


def move_Cursor(x,y,z):
    if z>=50:
        pyautogui.click()
        print("click")

    if (x > 0 and y == 0):  #moving to the right
        x, y = pyautogui.position()
        if (x + 40 <= 1920):
            pyautogui.moveTo(x+ 40, y, 0.2)
        else:
            return 0
    elif (x < 0 and y == 0):  # moving to the left
        x, y = pyautogui.position()
        if (x - 40 >= 0):
            pyautogui.moveTo(x - 40, y, 0.2)
        else:
            return 0
    elif (x == 0 and y > 0):  # moving down
        x, y = pyautogui.position()
        if (y + 40 <= 1080):
            pyautogui.moveTo(x, y + 40, 0.2)
        else:
            return 0

    elif (x == 0 and y < 0):  # moving up
        x, y = pyautogui.position()
        if (y - 40 >= 0):
            pyautogui.moveTo(x, y - 40, 0.2)
        else:
            return 0

    elif (x > 0 and y < 0):  # moving to the top right
        x, y = pyautogui.position()
        if (x + 40 <= 1920 and y - 40 >= 0):
            pyautogui.moveTo(x + 40, y - 40, 0.2)
        else:
            return 0

    elif (x > 0 and y > 0):  # moving to the bottom right
        x, y = pyautogui.position()
        if (x + 40 <= 1920 and y + 40 <= 1080):
            pyautogui.moveTo(x + 40, y + 40, 0.2)
        else:
            return 0

    elif (x < 0 and y < 0):  # moving to the top left
        x, y = pyautogui.position()
        if (x - 40 >= 0 and y - 40 >= 0):
            pyautogui.moveTo(x - 40, y - 40, 0.2)
        else:
            return 0

    elif (x < 0 and y > 0):  # moving to the bottom left
        x, y = pyautogui.position()
        if (x + 40 <= 1920 and y + 40 <= 1080):
            pyautogui.moveTo(x + 40, y + 40, 0.2)
        else:
            return 0

def cut_eyebrows(img):
    height, width = img.shape[:2]
    eyebrow_h = int(height / 4)
    img = img[eyebrow_h:height, 0:width]  # cut eyebrows out (15 px)
    return img


def blob_process(img, threshold, detector):
    _, img = cv.threshold(img, threshold, 255, cv.THRESH_BINARY)
    img = cv.erode(img, None, iterations=2)
    img = cv.dilate(img, None, iterations=4)
    img = cv.medianBlur(img, 5)
    cv.imshow('test 3', img)
    key_point = detector.detect(img)
    return key_point


def nothing(x):
    pass


def detectAndDisplay(frame):
    init_frame = frame
    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    frame_gray = cv.equalizeHist(frame_gray)
    # -- Detect faces
    faces = face_cascade.detectMultiScale(frame_gray)
    x_dir = 0
    y_dir = 0
    z = 1
    for (x, y, w, h) in faces:
        center = (x + w // 2, y + h // 2)
        frame = cv.ellipse(frame, center, (w // 2, h // 2), 0, 0, 360, (255, 0, 255), 4)
        faceROI = frame_gray[y:y + h, x:x + w]
        new_frame = init_frame[y:y + h, x:x + h]
        # -- In each face, detect eyes
        eyes = eyes_cascade.detectMultiScale(faceROI)
        # x_point, y_point = pyautogui.position() //add this later
        for (x2, y2, w2, h2) in eyes:
            z = 0
            img = new_frame[y2:y2 + h2, x2:x2 + w2]
            #cv.imshow('test2', img)  # show eye image capture for debugging
            threshold = cv.getTrackbarPos('threshold', 'Capture - Face detection')
            key_point = blob_process(img, threshold, detector)
            if len(key_point) > 0:
                print(w2 / 2, h2 / 2, key_point[0].pt[0], key_point[0].pt[1])
                if key_point[0].pt[0] < w2 / 2 * 0.9:
                    print('right')
                    x_move = 1
                elif key_point[0].pt[0] > w2 / 2 * 1.1:
                    print('left')
                    x_move = -1
                else:
                    print('middle')
                    x_move = 0

                if key_point[0].pt[1] > (h2/2) * 0.98:
                    print('down')
                    y_move = 1
                elif key_point[0].pt[1] < (h2 / 2) * 0.9:
                    print('up')
                    y_move = -1
                else:
                    print('middle')
                    y_move = 0
                x_dir += x_move
                y_dir += y_move
            frame = cv.rectangle(frame, (x + x2, y + y2), (x + x2 + w2, y + y2 + h2), (255, 60, 60), 2)

    cv.imshow('Capture - Face detection', frame)
    return x_dir, y_dir, z


def main():
    camera_device = args.camera
    cap = cv.VideoCapture(camera_device)
    cv.namedWindow('Capture - Face detection')
    if not cap.isOpened:
        print('--(!)Error opening video capture')
        exit(0)
    cv.createTrackbar('threshold', 'Capture - Face detection', 0, 255, nothing)
    non_eye = 0
    while True:
        ret, frame = cap.read()
        if frame is None:
            print('--(!) No captured frame -- Break!')
            break
        x, y , z=detectAndDisplay(frame)
        non_eye += z
        move_Cursor(x,y,non_eye)
        if non_eye >= 50 or x != 0 or y != 0:
            non_eye = 0
        if cv.waitKey(10) == 27:
            break


if __name__ == "__main__":
    main()