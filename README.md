# pic_diff
对两张图片进行对比，并将不一致的地方标识出来进行保存。

# 简介
1. 一共提供了两种图片对比方式：
    - 一种是基于OpenCV的图像识别，将图片二值化处理后进行对比，并将有差异的地方进行输出，一共会输出四张图片；
    - 另一种是基于百度飞桨OCR，主要用于文本的识别并进行对比，将不一致的地方进行标识后输出保存。
2. 在前期调研过程中，对市面上的OCR工具试用了几款，如pytesseract、百度OCR以及百度飞桨OCR。pytesseract的识别过程过慢，百度OCR使用的是开放API，每个月有免费的 1000 次调用额度，个人免费使用一下还好，不适合在项目中大量使用（有预算的除外）。百度的OCR是在我调研试用的几款工具中速度最快的，pytesseract识别一张图片大概需要 30s，百度ocr大概只需要2～3s，而飞桨的速度在两者之间大概在 10s 左右。所以综合考虑下来就使用了飞桨。

## 注意项：
1. 在`compare_images_paddle()`方法中设置了入参`offset`，之所以设置了该字段，是因为在两张图片识别完成后，相同位置的文字可能在结果的数组中的位置不一样。比如相同的一行字在图片 1 中的位置索引是 12，而在图片 2 中的位置却是 14。所以需要使用图片 1 的 12 位置的文字分别和图片 2 的 12 位置前后几个位置的文字分别进行对比。该字段默认是 2，也就是说会和 10、11、13、14 位置的文字均进行对比。
该值的设置可以根据各自实际使用情况分别进行修改，我的项目中 2 是一个合适的值，既不会增大误报率，也可以提高识别精度。
2. 在代码的 129 行～135 行，针对图片中的倾斜文字进行了过滤，这是为了减少我个人项目中用到的图片中带有的水印造成的误报。可以根据实际情况考虑是否需要，注释掉不影响使用。
