import cv2
import numpy as np

SHADOW_THRESHOLD = 100  # default 100
ALLOWED_SHADOW_PERCENTAGE = 10
filename = 'bad-thirts/shirt.jpg'
# filename = 'bad-thirts/red-tshirt.jpg'

IMG_WIDTH = 500
IMG_HEIGHT = 600


def getBlackPercentage(img):
    imgSize = img.shape[0] * img.shape[1]
    nonzero = cv2.countNonZero(img)
    return (imgSize - nonzero) * 100 / imgSize


img = cv2.imread(filename)

if img.shape[0] > IMG_HEIGHT or img.shape[1] > IMG_WIDTH:
    img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))  # Resize image

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, img_shadow_detected = cv2.threshold(gray, SHADOW_THRESHOLD, 255, 0)

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
