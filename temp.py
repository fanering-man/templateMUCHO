from re import A
import cv2
import numpy as np
import os
import glob
from PIL import Image


template_before = cv2.imread('A.jpg')
template_before_gray = cv2.cvtColor(template_before, cv2.COLOR_BGR2GRAY)

ROI = cv2.selectROI('Select ROIs', template_before, fromCenter = False, showCrosshair = False)
x1 = ROI[0]
y1 = ROI[1]
x2 = ROI[2]
y2 = ROI[3]
print('ROI', ROI)

template_after = template_before[int(y1):int(y1+y2),int(x1):int(x1+x2)]
template_after_gray = cv2.cvtColor(template_after, cv2.COLOR_BGR2GRAY)
cv2.imshow("ミニーマウス", template_after)
cv2.waitKey(0)

#BのAと同じROI部分
img = cv2.imread('B.jpg')
img2 = Image.open("B.jpg")
same_ROI= img2.crop((ROI[0],ROI[1],ROI[2]+ROI[0],ROI[3]+ROI[1])) 
same_ROI.save('./B_ROI.jpg')
same_ROI = cv2.imread('B_ROI.jpg')
same_ROI= cv2.cvtColor(same_ROI, cv2.COLOR_BGR2GRAY)

#分割-------------
# Aのtemplate_after_grayを分割した→chunk
from pathlib import Path

rows = 4  # 行数
cols = 4  # 列数

chunks = []
for row_img in np.array_split(template_after_gray, rows, axis=0):
    for chunk in np.array_split(row_img, cols, axis=1):
        chunks.append(chunk)
print("枚数",len(chunks))

output_dir = Path("output")
output_dir.mkdir(exist_ok=True)
for i, chunk in enumerate(chunks):
    save_path = output_dir / f"chunk_{i:02d}.png"
    cv2.imwrite(str(save_path), chunk)
    # cv2.imshow(f"chunk{i}",chunk)
    # cv2.waitKey(700)
#------------------



    # 処理対象画像に対して、テンプレート画像との類似度を算出する
    res = cv2.matchTemplate(same_ROI,chunk, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)


    print(i,"   枚目!!!")
    print("max_val:",max_val)
    print("Max res :",np.ndarray.max(res))
    print("maxloc :",max_loc)

    # テンプレートマッチング画像の高さ、幅を取得する
    hw = chunk.shape
    print("hw :",hw)

    # 検出した部分に赤枠をつける
    cv2.rectangle(img, (max_loc[0]+ROI[0],max_loc[1]+ROI[1] ) , (max_loc[0]+hw[1]+ROI[0],max_loc[1]+hw[0]+ROI[1]),(0, 0, 255), 2)#第三引数のhw → w,hの順番

# # 画像の保存
# # cv2.imwrite('./tpl_match_after.png', img)
cv2.imshow("うんち",img)
cv2.waitKey()
