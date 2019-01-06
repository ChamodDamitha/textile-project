import cv2
import numpy as np
import imutils

ALLOWED_SHADOW_PERCENTAGE = 10
filename = 'bad-thirts/shirt.jpg'
# filename = 'bad-thirts/red-tshirt.jpg'
filename = 'bad-thirts/black-tshirt.jpeg'
# filename = 'bad-thirts/pink-tshirt.jpeg'
# filename = 'bad-thirts/black-tshirt-2.jpg'
# filename = 'bad-thirts/tshirt-backgound-2.jpg'
# filename = 'bad-thirts/messy-background-tshirt.jpg'

IMG_WIDTH = 500
IMG_HEIGHT = 600


def getBlackPercentage(img):
    imgSize = img.shape[0] * img.shape[1]
    nonzero = cv2.countNonZero(img)
    return (imgSize - nonzero) * 50 / imgSize


img = cv2.imread(filename)

if img.shape[0] > IMG_HEIGHT or img.shape[1] > IMG_WIDTH:
    img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))  # Resize image

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Use color histogram to find dominant colors
hist = cv2.calcHist([gray], [0], None, [256], [0, 256])

# Convert histogram to simple list
hist = [val[0] for val in hist];

# Generate a list of indices
indices = list(range(0, 256));

# Descending sort-by-key with histogram value as key
s = [(x, y) for y, x in sorted(zip(hist, indices), reverse=True)]

max_points = []
prev = -1
for i in range(len(hist)):
    if i + 1 < len(hist) and hist[i + 1] < hist[i] and \
            hist[i] > prev and i != gray[0][0]:  # assume (0,0) color = background color
        max_points.append(i)
    prev = hist[i]

# Find 2 highest peak points except background color
peak_points = []
c = 0
for v in s:
    if v[0] in max_points:
        c += 1
        peak_points.append(v[0])
        if c == 2: break

shadow_threshold = np.mean(peak_points)

# cv2.imshow('Gray', gray)
# edged = cv2.Canny(gray, 30, 100)
# # cv2.imshow('Edged', edged)
# cnts = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# cnts = imutils.grab_contours(cnts)
# cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:10]
# cv2.drawContours(img, cnts, -1, (0, 255, 0), 3)
# cv2.imshow("Img", img)
#
# # flood fill background
# queue = [(0, 0)]
# map = np.zeros(gray.shape)
# while len(queue) > 0:
#     p = queue.pop()
#     if img[p[0]][p[1]][0] == 0 and img[p[0]][p[1]][1] == 255 and img[p[0]][p[1]][2] == 0 :
#         gray[p[0]][p[1]] = 255
#     map[p[0]][p[1]] = 1
#     if p[0] + 1 < gray.shape[0]:
#         if map[p[0] + 1][p[1]] == 0: queue.append((p[0] + 1, p[1]))
#         if p[1] - 1 > 0:
#             if map[p[0] + 1][p[1] - 1] == 0: queue.append((p[0] + 1, p[1] - 1))
#         if p[1] + 1 < gray.shape[1]:
#             if map[p[0] + 1][p[1] + 1] == 0: queue.append((p[0] + 1, p[1] + 1))
#     if p[0] - 1 > 0:
#         if map[p[0] - 1][p[1]] == 0: queue.append((p[0] - 1, p[1]))
#         if p[1] - 1 > 0:
#             if map[p[0] - 1][p[1] - 1] == 0: queue.append((p[0] - 1, p[1] - 1))
#         if p[1] + 1 < gray.shape[1]:
#             if map[p[0] - 1][p[1] + 1] == 0: queue.append((p[0] - 1, p[1] + 1))
#     if p[1] - 1 > 0:
#         if map[p[0]][p[1] - 1] == 0: queue.append((p[0], p[1] - 1))
#     if p[1] + 1 < gray.shape[1]:
#         if map[p[0]][p[1] + 1] == 0: queue.append((p[0], p[1] + 1))

# cv2.imshow('Gray', gray)


print("Threshold color : " + str(shadow_threshold))
print("Threshold Black % : " + str(ALLOWED_SHADOW_PERCENTAGE))

ret, img_shadow_detected = cv2.threshold(gray, shadow_threshold, 255, 0)

img_shadow_detected_3d = img.copy()
arr_up = img_shadow_detected > 128
arr_down = img_shadow_detected <= 128
img_shadow_detected_3d[arr_up] = [255, 255, 255]
img_shadow_detected_3d[arr_down] = [0, 0, 0]

combined_img = np.concatenate((img, img_shadow_detected_3d), axis=1)
cv2.imshow('Output', combined_img)

black_percentage = getBlackPercentage(img_shadow_detected)
print("Black % : " + str(black_percentage))

if black_percentage > ALLOWED_SHADOW_PERCENTAGE:
    print("Status : BAD")
else:
    print("Status : GOOD")

if cv2.waitKey(0) & 0xff == 27:
    cv2.destroyAllWindows()
