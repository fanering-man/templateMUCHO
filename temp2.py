import cv2
import numpy as np


# 画像の読み込み + グレースケール化
img = cv2.imread('A.jpg')
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

template_before = cv2.imread('B.jpg')
template_before_gray = cv2.cvtColor(template_before, cv2.COLOR_BGR2GRAY)


ROI = cv2.selectROI('Select ROIs', template_before, fromCenter = False, showCrosshair = False)
x1 = ROI[0]
y1 = ROI[1]
x2 = ROI[2]
y2 = ROI[3]
print('ROI', ROI)
#Crop Image
template_after = template_before[int(y1):int(y1+y2),int(x1):int(x1+x2)]
template_after_gray = cv2.cvtColor(template_after, cv2.COLOR_BGR2GRAY)
cv2.imshow("ミニーマウス", template_after)
cv2.waitKey(0)

# template_before_gray = cv2.cvtColor(template_before, cv2.COLOR_BGR2GRAY)

# 処理対象画像に対して、テンプレート画像との類似度を算出する
res = cv2.matchTemplate(img_gray,template_after_gray, cv2.TM_CCOEFF_NORMED)
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

# 類似度の高い部分を検出する
# threshold = 0.7
# loc = np.where(np.ndarray.max(res))
print("res :",res)
print("max_val:",max_val)
print("Maxres :",np.ndarray.max(res))
# loc = np.ndarray.max(res)
# print("loc :",loc)
print("maxloc :",max_loc)
# テンプレートマッチング画像の高さ、幅を取得する
hw = template_after_gray.shape
print("hw :",hw)

# 検出した部分に赤枠をつける
# for pt in zip(*loc[::-1]):
cv2.rectangle(img, max_loc  , (max_loc[0]+hw[1],max_loc[1]+hw[0]),(0, 0, 255), 2)#w,h
# # 画像の保存
# # cv2.imwrite('./tpl_match_after.png', img)
cv2.imshow("a",img)
cv2.waitKey()
