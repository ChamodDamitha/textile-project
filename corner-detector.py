import cv2
import numpy as np
import math
import csv
import imutils

# in cm
CAMERA_HEIGHT = 136
LENGTH_TO_PIXEL_AND_CAMERA_HEIGHT_RATIO = 94.0 / (1000 * 136)


def setMeasurements(measurements):
    with open("measurements.csv", mode='w') as measurements_file:
        measurements['ERROR'] = input("Enter error in cm : ")
        measurements['WIDTH'] = input("Enter width in cm : ")
        measurements['HEIGHT'] = input("Enter height in cm : ")

        measurements_writer = csv.writer(measurements_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

        measurements_writer.writerow(['ERROR', measurements['ERROR']])
        measurements_writer.writerow(['WIDTH', measurements['WIDTH']])
        measurements_writer.writerow(['HEIGHT', measurements['HEIGHT']])

        print("Required measurements(cm).....")
        print(measurements)

    return measurements


measurements = {}

try:
    with open("measurements.csv", mode='r') as measurements_file:
        measurements_reader = csv.reader(measurements_file)
        for row in measurements_reader:
            measurements[row[0]] = row[1]

        if len(measurements) < 3:
            measurements = setMeasurements(measurements)
        else:
            print("Required measurements(cm).....")
            print(measurements)
            SET_MEASUREMENTS = raw_input("Change current measurements? [N/y]")
            if SET_MEASUREMENTS.lower() == 'y':
                measurements = setMeasurements(measurements)
except IOError:
    measurements = setMeasurements(measurements)

filename = 'good-thirts/real-tshirt-new.jpeg'
img = cv2.imread(filename)
img = cv2.resize(img, (1000, 1000))
cv2.imshow("Show by CV2", img)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

(thresh, im_bw) = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

# im_bw = cv2.blur(im_bw, (5, 5))

im_bw = np.float32(im_bw)
dst = cv2.cornerHarris(im_bw, 2, 3, 0.03)

# result is dilated for marking the corners, not important
dst = cv2.dilate(dst, None)

threshold = 0.1 * dst.max()
# Threshold for an optimal value, it may vary depending on the image.
arr = dst > threshold
img[arr] = [0, 255, 0]

corners = []


def addCorner(i, j):
    t = 15
    for c in corners:
        if i > c[0] + t or i < c[0] - t or j > c[1] + t or j < c[1] - t:
            continue
        else:
            return
    corners.append((i, j))
    return


def length(x, y):
    return ((x[0] - y[0]) ** 2 + (x[1] - y[1]) ** 2) ** 0.5


def distance(m, n, point):
    a = m[1] - n[1]
    b = n[0] - m[0]
    c = (m[0] * n[1]) - (n[0] * m[1])
    return math.fabs((a * point[0] + b * point[1] + c) / ((a * a + b * b) ** 0.5))


for j in range(len(arr)):
    for i in range(len(arr[j])):
        if arr[j][i]:
            addCorner(i, j)

corners.sort(key=lambda t: t[1])

colar = corners[0]
# shoulders = corners[2:4]


bottom_corners = [c for c in corners if c[1] > img.shape[1] / 2]
bottom_corners.sort(key=lambda t:t[0])
bottoms = [bottom_corners[0], bottom_corners[-1]]

# print("shoulders", shoulders)
print
print("Current Measurements(cm)...........")

# print("shoulder width : ", length(shoulders[0], shoulders[1]))
width = length(bottoms[0], bottoms[1]) * LENGTH_TO_PIXEL_AND_CAMERA_HEIGHT_RATIO * CAMERA_HEIGHT
height = distance(bottoms[0], bottoms[1], colar) * LENGTH_TO_PIXEL_AND_CAMERA_HEIGHT_RATIO * CAMERA_HEIGHT
print("width(cm) : " + str(width))
print("height(cm) : " + str(height))

if math.fabs(width - int(measurements['WIDTH'])) < int(measurements['ERROR']) \
        and math.fabs(height - int(measurements['HEIGHT'])) < int(measurements['ERROR']):
    print("STATUS : GOOD")
else:
    print("STATUS : BAD")

# cv2.imshow('corner', im_bw)
# cv2.imshow('corner-detected', cv2.blur(im_bw, (5, 5)))

cv2.imshow('corner-detected', img)

if cv2.waitKey(0) & 0xff == 27:
    cv2.destroyAllWindows()
