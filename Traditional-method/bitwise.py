import numpy as np
import cv2

'''本文目的：使用opencv套件，簡單模擬傳統AOI檢測方式'''

#img1為良品圖，img2為瑕庛圖
img1 = cv2.imread('./PCBData/group00041/00041/00041010_temp.jpg',0)
img2 = cv2.imread('./PCBData/group00041/00041/00041010_test.jpg',0)


#先將良品和瑕疵圖用XOR二值法，顯示出差異圖
bitwiseXor = cv2.bitwise_xor(img1, img2)
cv2.imshow("XOR", bitwiseXor)


#由於差異圖中，有邊框存在，需將邊框去除
#先用Canny取出良品圖邊框
edges = cv2.Canny(img1,100,200)
cv2.imshow("Canny",edges)


#將差異圖減去邊框，再用影象平滑中的中值模糊去除雜值，顯示瑕疵點：Final圖
# Final = cv2.absdiff(bitwiseXor,edges)
Final = cv2.subtract(bitwiseXor,edges)

# Final = cv2.GaussianBlur(Final, (7,7), 0)
# Final = cv2.blur(Final, (3, 3))
Final = cv2.medianBlur(Final, 5)
cv2.imshow("Final", Final)


cv2.waitKey(0)

cv2.destroyAllWindows()
