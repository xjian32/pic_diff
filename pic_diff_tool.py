"""
@Project : Project
@File    : pic_diff_tool.py
@IDE     : PyCharm 
@Date    : 2024/8/26 09:40 
@Author  : Jaxx
@DESC    : 图片对比功能
"""
from pathlib import Path
from typing import Union, Optional, Tuple

from skimage.metrics import structural_similarity

from paddleocr import PaddleOCR
from PIL import Image, ImageDraw
import imutils
import cv2
import numpy as np
import arrow

from loguru import logger


def resize(img1: str, img2: str) -> Tuple:
    """
    将两张图片调整为相同尺寸
    :param img1:
    :param img2:
    :return:
    """
    imageA = cv2.imread(img1)
    imageB = cv2.imread(img2)
    # 获取图片尺寸
    size1 = imageA.shape[:2]
    size2 = imageB.shape[:2]

    # 通过比较面积，选择较小的尺寸作为目标尺寸
    if size1[0] * size1[1] < size2[0] * size2[1]:
        target_size = (size1[1], size1[0])
    else:
        target_size = (size2[1], size2[0])

    return (
        cv2.resize(imageA, target_size, interpolation=cv2.INTER_AREA),
        cv2.resize(imageB, target_size, interpolation=cv2.INTER_AREA),
    )


def pic_diff_cv(
    pic1: str,
    pic2: str,
    output_path: Optional[Union[str, Path]] = None,
    show_time: int = 2000,
) -> bool:
    """
    对比两张图片，并生成原图的标识图、差异图
    :param show_time: 差异图片显示时长，默认 2000ms，如果不需要显示，则设置为 0
    :param output_path: 对比图片输出路径
    :param pic1: 图片 1 的地址
    :param pic2: 图片 2 的地址
    :return: bool
    """
    resized_image1, resized_image2 = resize(pic1, pic2)

    # 判断两张图片是否一致
    difference = cv2.subtract(resized_image1, resized_image2)
    result = not np.any(difference)
    if result is True:
        logger.info(f"📢两张图片相同")
        return True
    grayA = cv2.cvtColor(resized_image1, cv2.COLOR_BGR2GRAY)
    grayB = cv2.cvtColor(resized_image2, cv2.COLOR_BGR2GRAY)

    (score, diff) = structural_similarity(grayA, grayB, full=True)
    diff = (diff * 255).astype("uint8")

    logger.warning(f"⚠️两张图片不一样，结构相似性指数 SSIM: [{score:.5f}]")
    thresh = cv2.threshold(diff, 128, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    # 遍历绘制轮廓
    for c in cnts:
        (x, y, w, h) = cv2.boundingRect(c)
        cv2.rectangle(resized_image1, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.rectangle(resized_image2, (x, y), (x + w, y + h), (0, 0, 255), 2)
    if show_time != 0:
        cv2.imshow("图片1", resized_image1)
        cv2.waitKey(show_time)

    if output_path is None:
        output_path = "./"
    if isinstance(output_path, Path):
        output_path = str(output_path)
    logger.info(f"图片对比结果已保存到：{output_path}")
    _tmp = arrow.now().format("YYYYMMDD-HHmmss")
    cv2.imwrite(f"{output_path}/标识图-1-{_tmp}.jpg", resized_image1)
    cv2.imwrite(f"{output_path}/标识图-2-{_tmp}.jpg", resized_image2)
    cv2.imwrite(f"{output_path}/反转差异图-{_tmp}.jpg", thresh)
    cv2.imwrite(f"{output_path}/差异图-{_tmp}.jpg", diff)

    cv2.destroyAllWindows()
    return False


def compare_images_paddle(
    image1_path: str,
    image2_path: str,
    offset: int = 2,
    output_path: Optional[Union[str, Path]] = None,
) -> int:
    """
    使用飞桨OCR进行图片识别对比
    :param image1_path: 图片一的路径
    :param image2_path: 图片二的路径
    :param offset: 容错值，默认为 2
    :param output_path: 图片对比结果输出路径
    :return: 图片差异数
    """
    ocr = PaddleOCR(
        use_angle_cls=True, lang="ch"
    )  # need to run only once to download and load model into memory
    result1 = ocr.ocr(image1_path, cls=False)
    result2 = ocr.ocr(image2_path, cls=False)

    img1 = Image.open(image1_path)
    img2 = Image.open(image2_path)
    draw1 = ImageDraw.Draw(img1)
    draw2 = ImageDraw.Draw(img2)

    result2 = result2[0]
    # print(result1[0])
    diff_num = 0
    for i in range(len(result1 := result1[0])):
        found = False
        if (
            result1[i][0][0][1] != result1[i][0][1][1]
            or result2[i][0][0][1] != result2[i][0][1][1]
        ):
            # 过滤非水平方向的文字，在我项目中为水印内容
            # 实际使用时根据情况是否注释掉
            continue
        if i < len(result2):
            if result1[i][1][0] == result2[i][1][0]:
                found = True
                continue

            if found is False:
                for _offset in [i for i in range(-offset, offset + 1) if i != 0]:
                    checked_index = i + _offset
                    if (
                        0 <= checked_index < len(result2)
                        and result1[i][1][0] == result2[checked_index][1][0]
                    ):
                        found = True
                        break
            if found is False:
                # 获取位置
                point1 = result1[i][0]
                left1 = min(point1[0][0], point1[1][0], point1[2][0], point1[3][0])
                top1 = min(point1[0][1], point1[1][1], point1[2][1], point1[3][1])
                right1 = max(point1[0][0], point1[1][0], point1[2][0], point1[3][0])
                bottom1 = max(point1[0][1], point1[1][1], point1[2][1], point1[3][1])

                point2 = result2[i][0]
                left2 = min(point2[0][0], point2[1][0], point2[2][0], point2[3][0])
                top2 = min(point2[0][1], point2[1][1], point2[2][1], point2[3][1])
                right2 = max(point2[0][0], point2[1][0], point2[2][0], point2[3][0])
                bottom2 = max(point2[0][1], point2[1][1], point2[2][1], point2[3][1])

                draw1.rectangle((left1, top1, right1, bottom1), outline="red", width=2)
                draw2.rectangle((left2, top2, right2, bottom2), outline="red", width=2)

                diff_num += 1

    if output_path is None:
        output_path = "./"
    if isinstance(output_path, Path):
        output_path = str(output_path)
    img1.save(f"{output_path}/paddle_img1.png")
    img2.save(f"{output_path}/paddle_img2.png")
    logger.info(f"图片对比完成，结果保存在: {output_path}，此次不一致数为：{diff_num}")

    return diff_num


if __name__ == "__main__":
    img1 = "./Pictures/测试图片 01.jpeg"
    img2 = "./Downloads/下载.jpeg"

    resize(img1, img2)
    pic_diff_cv(img1, img2)

    compare_images_paddle(img1, img2, 2)
