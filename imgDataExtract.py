import cv2
import easyocr

# 创建 OCR reader
reader = easyocr.Reader(['ch_sim', 'en'])  # 使用简体中文和英文识别

# 读取图片
image_path = "./health.jpeg"
img = cv2.imread(image_path)

# 转换为灰度图像
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 应用二值化处理
threshold_img = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

# 使用 OCR 识别文本
result = reader.readtext(threshold_img)

# 提取识别结果的文本部分
text = "\n".join([item[1] for item in result])

print("识别结果：")
print(text)





