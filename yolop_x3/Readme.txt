需求：yolop模型部署至x3派 调用mipi相机web端显示

文件：
1. yolop_x3.py : 可以部署至x3派进行图片的推理
2. yolop_x3_mipicamera.py : 对mipi相机输出的视频流进行vps缩小  mipi端显示，但是画面报错(可能不支持640*640显示)
3. yolop_x3_mipicamera_web.py : 有画面，但是没有推理结果