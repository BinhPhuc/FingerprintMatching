import os
import cv2
import numpy as np
import math

def mse(img1, img2):
    height, width = img1.shape
    gap = cv2.subtract(img1, img2)
    error = np.sum(gap ** 2)
    mse = error / (float(height * width))
    return mse, gap

goal = cv2.imread('SOCOFING/Altered/Altered-Hard/150__M_Right_index_finger_Obl.BMP')
goal = cv2.cvtColor(goal, cv2.COLOR_BGR2GRAY)

best_score = math.inf
fileName = None
gap_img = None

for file in os.listdir('SOCOFING/Real'):
   finger_print_img = cv2.imread('SOCOFING/Real/' + file)
   finger_print_img = cv2.cvtColor(finger_print_img, cv2.COLOR_BGR2GRAY)
   if(finger_print_img.shape != goal.shape):
       finger_print_img = cv2.resize(finger_print_img, (goal.shape[1], goal.shape[0]))
   score, gap = mse(goal, finger_print_img)
   if(score < best_score):
       best_score = score
       fileName = file
       gap_img = gap
       
print("Matching with file name: " + fileName)
print("Best score: " + str(best_score))
cv2.imwrite('output/gap_img.BMP', gap)
matching_img = cv2.imread('SOCOFING/Real/' + fileName)
cv2.imwrite('output/matching_img.BMP', matching_img)
cv2.imshow("gap", gap_img)
cv2.waitKey(0)
cv2.destroyAllWindows()