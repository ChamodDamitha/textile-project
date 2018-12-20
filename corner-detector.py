import cv2
import numpy as np
import math

filename = 'good-thirts/tshirt-sample.png'
img = cv2.imread(filename)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

(thresh, im_bw) = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

# im_bw = cv2.blur(im_bw, (5, 5))

im_bw = np.float32(im_bw)
dst = cv2.cornerHarris(im_bw, 2, 3, 0.04)

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

print("corners", corners)

corners.sort(key=lambda t: t[1])
print("corners", corners)

shoulders = corners[2:4]
bottoms = corners[-2:]

print("shoulders", shoulders)
print("bottoms", bottoms)
print
print
print("Measurements...........")

print("shoulder width : ", length(shoulders[0], shoulders[1]))
print("width : ", length(bottoms[0], bottoms[1]))
print("height : ", distance(bottoms[0], bottoms[1], shoulders[0]))

# cv2.imshow('corner', im_bw)
# cv2.imshow('corner-detected', cv2.blur(im_bw, (5, 5)))
cv2.imshow('corner-detected', img)
if cv2.waitKey(0) & 0xff == 27:
    cv2.destroyAllWindows()
