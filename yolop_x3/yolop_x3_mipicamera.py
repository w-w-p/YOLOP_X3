import sys
import signal
import os
import numpy as np
import cv2
import colorsys
import argparse
from time import time,sleep
import multiprocessing
from threading import BoundedSemaphore
import ctypes
import json 
import torch
# Camera API libs
from lib.core.general import non_max_suppression
from hobot_vio import libsrcampy as srcampy
from hobot_dnn import pyeasy_dnn as dnn
import threading

# 定义结构体类
class hbSysMem_t(ctypes.Structure):
    _fields_ = [
        ("phyAddr", ctypes.c_double),
        ("virAddr", ctypes.c_void_p),
        ("memSize", ctypes.c_int)
    ]

class hbDNNQuantiShift_t(ctypes.Structure):
    _fields_ = [
        ("shiftLen", ctypes.c_int),
        ("shiftData", ctypes.c_char_p)
    ]

class hbDNNQuantiScale_t(ctypes.Structure):
    _fields_ = [
        ("scaleLen", ctypes.c_int),
        ("scaleData", ctypes.POINTER(ctypes.c_float)),
        ("zeroPointLen", ctypes.c_int),
        ("zeroPointData", ctypes.c_char_p)
    ]

class hbDNNTensorShape_t(ctypes.Structure):
    _fields_ = [
        ("dimensionSize", ctypes.c_int * 8),
        ("numDimensions", ctypes.c_int)
    ]

class hbDNNTensorProperties_t(ctypes.Structure):
    _fields_ = [
        ("validShape", hbDNNTensorShape_t),
        ("alignedShape", hbDNNTensorShape_t),
        ("tensorLayout", ctypes.c_int),
        ("tensorType", ctypes.c_int),
        ("shift", hbDNNQuantiShift_t),
        ("scale", hbDNNQuantiScale_t),
        ("quantiType", ctypes.c_int),
        ("quantizeAxis", ctypes.c_int),
        ("alignedByteSize", ctypes.c_int),
        ("stride", ctypes.c_int * 8)
    ]

class hbDNNTensor_t(ctypes.Structure):
    _fields_ = [
        ("sysMem", hbSysMem_t * 4),
        ("properties", hbDNNTensorProperties_t)
    ]

class Yolov5PostProcessInfo_t(ctypes.Structure):
    _fields_ = [
        ("height", ctypes.c_int),
        ("width", ctypes.c_int),
        ("ori_height", ctypes.c_int),
        ("ori_width", ctypes.c_int),
        ("score_threshold", ctypes.c_float),
        ("nms_threshold", ctypes.c_float),
        ("nms_top_k", ctypes.c_int),
        ("is_pad_resize", ctypes.c_int)
    ]

libpostprocess = ctypes.CDLL('/usr/lib/libpostprocess.so')

get_Postprocess_result = libpostprocess.Yolov5PostProcess
get_Postprocess_result.argtypes = [ctypes.POINTER(Yolov5PostProcessInfo_t)]
get_Postprocess_result.restype = ctypes.c_char_p 

def get_TensorLayout(Layout):
    if Layout == "NCHW":
        return int(2)
    else:
        return int(0)
def get_hw(pro):
    if pro.layout == "NCHW":
        return pro.shape[2], pro.shape[3]
    else:
        return pro.shape[1], pro.shape[2]


# def nv12_to_bgr(image_nv12, width, height):
#     y_size = width * height
#     uv_size = y_size // 2
#     y = image_nv12[:y_size].reshape((height, width))
#     uv = image_nv12[y_size:].reshape((height // 2, width)).reshape((height // 2, width, 2))
#     yuv = np.zeros((height + height // 2, width), dtype=np.uint8)
#     yuv[:height, :] = y
#     yuv[height:, :] = uv.reshape((height // 2, width * 2))
#     return cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_NV12)

def nv12_to_bgr(image_nv12, width, height):
    yuvdata = np.frombuffer(image_nv12, dtype=np.uint8)
    bgr_img = cv2.cvtColor(yuvdata.reshape((height * 3 // 2, width)), cv2.COLOR_YUV2BGR_NV12)
    return bgr_img

def bgr2nv12_opencv(image):
    height, width = image.shape[0], image.shape[1]
    area = height * width
    
    yuv420p = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_BGR2YUV_I420).reshape((area * 3 // 2,))
    y = yuv420p[:area]
    uv_planar = yuv420p[area:].reshape((2, area // 4))
    uv_packed = uv_planar.transpose((1, 0)).reshape((area // 2,))

    nv12 = np.zeros_like(yuv420p)
    nv12[:height * width] = y
    nv12[height * width:] = uv_packed
    return nv12

def resize_unscale(img, new_shape=(640, 640), color=114):
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    canvas = np.zeros((new_shape[0], new_shape[1], 3))
    canvas.fill(color)
    # Scale ratio (new / old) new_shape(h,w)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])

    # Compute padding
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))  # w,h
    new_unpad_w = new_unpad[0]
    new_unpad_h = new_unpad[1]
    pad_w, pad_h = new_shape[1] - new_unpad_w, new_shape[0] - new_unpad_h  # wh padding

    dw = pad_w // 2  # divide padding into 2 sides
    dh = pad_h // 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_AREA)

    canvas[dh:dh + new_unpad_h, dw:dw + new_unpad_w, :] = img

    return canvas, r, dw, dh, new_unpad_w, new_unpad_h  # (dw,dh)

def print_properties(pro):
    print("tensor type:", pro.tensorType)
    print("data type:", pro.tensorType)
    print("layout:", pro.tensorLayout)
    print("shape:", [pro.validShape.dimensionSize[i] for i in range(pro.validShape.numDimensions)])

def signal_handler(signal, frame):
    global is_stop
    print("Stopping!\n")
    is_stop = True

def infer_yolop(weight,img_path):

    model_path = weight
    model = pyeasy_dnn.load(model_path)
    print(f"Load {model_path} done!")

    save_det_path = f"./pictures/detect_onnx.jpg"
    save_da_path = f"./pictures/da_onnx.jpg"
    save_ll_path = f"./pictures/ll_onnx.jpg"
    save_merge_path = f"./pictures/output_onnx.jpg"

    img_bgr = cv2.imread(img_path)
    height, width, _ = img_bgr.shape

    img0 = img_bgr.copy().astype(np.uint8)
    img_rgb = img_bgr[:, :, ::-1].copy()
    
    h, w = get_hw(model[0].inputs[0].properties)

    canvas, r, dw, dh, new_unpad_w, new_unpad_h = resize_unscale(img0, (h, w))
    img_input = bgr2nv12_opencv(canvas)

    preds = model[0].forward(img_input)

    det_out = preds[0].buffer[...,0]
    da_seg_out = preds[1].buffer
    ll_seg_out = preds[2].buffer

    det_out = torch.from_numpy(det_out).float()
    boxes = non_max_suppression(det_out)[0]  # [n,6] [x1,y1,x2,y2,conf,cls]
    boxes = boxes.cpu().numpy().astype(np.float32)

    if boxes.shape[0] == 0:
        print("no bounding boxes detected.")
        return

    # scale coords to original size.
    boxes[:, 0] -= dw
    boxes[:, 1] -= dh
    boxes[:, 2] -= dw
    boxes[:, 3] -= dh
    boxes[:, :4] /= r

    print(f"detect {boxes.shape[0]} bounding boxes.")

    img_det = img_rgb[:, :, ::-1].copy()
    for i in range(boxes.shape[0]):
        x1, y1, x2, y2, conf, label = boxes[i]
        x1, y1, x2, y2, label = int(x1), int(y1), int(x2), int(y2), int(label)
        img_det = cv2.rectangle(img_det, (x1, y1), (x2, y2), (0, 255, 0), 2, 2)

    cv2.imwrite(save_det_path, img_det)

    # select da & ll segment area.
    da_seg_out = da_seg_out[:, :, dh:dh + new_unpad_h, dw:dw + new_unpad_w]
    ll_seg_out = ll_seg_out[:, :, dh:dh + new_unpad_h, dw:dw + new_unpad_w]

    da_seg_mask = np.argmax(da_seg_out, axis=1)[0]  # (?,?) (0|1)
    ll_seg_mask = np.argmax(ll_seg_out, axis=1)[0]  # (?,?) (0|1)
    print(da_seg_mask.shape)
    print(ll_seg_mask.shape)

    color_area = np.zeros((new_unpad_h, new_unpad_w, 3), dtype=np.uint8)
    color_area[da_seg_mask == 1] = [0, 255, 0]
    color_area[ll_seg_mask == 1] = [255, 0, 0]
    color_seg = color_area

    # convert to BGR
    color_seg = color_seg[..., ::-1]
    color_mask = np.mean(color_seg, 2)
    img_merge = canvas[dh:dh + new_unpad_h, dw:dw + new_unpad_w, :]
    img_merge = img_merge[:, :, ::-1]

    # merge: resize to original size
    img_merge[color_mask != 0] = \
        img_merge[color_mask != 0] * 0.5 + color_seg[color_mask != 0] * 0.5
    img_merge = img_merge.astype(np.uint8)
    img_merge = cv2.resize(img_merge, (width, height),
                           interpolation=cv2.INTER_LINEAR)
    for i in range(boxes.shape[0]):
        x1, y1, x2, y2, conf, label = boxes[i]
        x1, y1, x2, y2, label = int(x1), int(y1), int(x2), int(y2), int(label)
        img_merge = cv2.rectangle(img_merge, (x1, y1), (x2, y2), (0, 255, 0), 2, 2)

    # da: resize to original size
    da_seg_mask = da_seg_mask * 255
    da_seg_mask = da_seg_mask.astype(np.uint8)
    da_seg_mask = cv2.resize(da_seg_mask, (width, height),
                             interpolation=cv2.INTER_LINEAR)

    # ll: resize to original size
    ll_seg_mask = ll_seg_mask * 255
    ll_seg_mask = ll_seg_mask.astype(np.uint8)
    ll_seg_mask = cv2.resize(ll_seg_mask, (width, height),
                             interpolation=cv2.INTER_LINEAR)

    cv2.imwrite(save_merge_path, img_merge)
    cv2.imwrite(save_da_path, da_seg_mask)
    cv2.imwrite(save_ll_path, ll_seg_mask)

    print("detect done.")

def infer_yolop_camera(weight,image_nv12):
    model_path = weight
    model = dnn.load(model_path)
    print(f"Load {model_path} done!")

    # height = int(image_nv12.shape[0] * 2 / 3)  # 根据NV12格式计算高度
    # width = image_nv12.shape[1]
    try:
        imgbgr = nv12_to_bgr(image_nv12, 640, 640)
    except Exception as e:
        print(f"Failed to nv12_to_bgr : {e}")
        exit(1) 

    img_input = image_nv12
    # 代码使用模型对输入图像进行推理  取模型的输出
    preds = model[0].forward(img_input)
    # 从模型输出中提取检测结果、动态区域分割结果和车道线分割结果
    det_out = preds[0].buffer[...,0]
    da_seg_out = preds[1].buffer
    ll_seg_out = preds[2].buffer
    # 将检测结果转换为 PyTorch 张量，并使用非极大值抑制（NMS）来过滤多余的框
    # 将检测框的坐标转换回原图尺寸
    det_out = torch.from_numpy(det_out).float()
    boxes = non_max_suppression(det_out)[0]  # [n,6] [x1,y1,x2,y2,conf,cls]
    boxes = boxes.cpu().numpy().astype(np.float32)
    # 如果没有检测到任何框，则输出消息并返回原始图像
    if boxes.shape[0] == 0:
        print("no bounding boxes detected.")
        return imgbgr
    # 将检测框的坐标缩放回原始图像尺寸
    # scale coords to original size.
    boxes[:, 0] -= dw
    boxes[:, 1] -= dh
    boxes[:, 2] -= dw
    boxes[:, 3] -= dh
    boxes[:, :4] /= r

    print(f"detect {boxes.shape[0]} bounding boxes.")
    # 在图像上绘制检测框，并返回带有检测框的图像
    img_det = img_rgb[:, :, ::-1].copy()
    for i in range(boxes.shape[0]):
        x1, y1, x2, y2, conf, label = boxes[i]
        x1, y1, x2, y2, label = int(x1), int(y1), int(x2), int(y2), int(label)
        img_det = cv2.rectangle(img_det, (x1, y1), (x2, y2), (0, 255, 0), 2, 2)
    
    # return img_det
    da_seg_out = da_seg_out[:, :, dh:dh + new_unpad_h, dw:dw + new_unpad_w]
    ll_seg_out = ll_seg_out[:, :, dh:dh + new_unpad_h, dw:dw + new_unpad_w]

    da_seg_mask = np.argmax(da_seg_out, axis=1)[0]
    ll_seg_mask = np.argmax(ll_seg_out, axis=1)[0]
    print(da_seg_mask.shape)
    print(ll_seg_mask.shape)

    color_area = np.zeros((new_unpad_h, new_unpad_w, 3), dtype=np.uint8)
    color_area[da_seg_mask == 1] = [0, 255, 0]
    color_area[ll_seg_mask == 1] = [255, 0, 0]
    color_seg = color_area

    color_seg = color_seg[..., ::-1]
    color_mask = np.mean(color_seg, 2)
    img_merge = canvas[dh:dh + new_unpad_h, dw:dw + new_unpad_w, :]
    img_merge = img_merge[:, :, ::-1]

    img_merge[color_mask != 0] = img_merge[color_mask != 0] * 0.5 + color_seg[color_mask != 0] * 0.5
    img_merge = img_merge.astype(np.uint8)
    img_merge = cv2.resize(img_merge, (width, height), interpolation=cv2.INTER_LINEAR)

    for i in range(boxes.shape[0]):
        x1, y1, x2, y2, conf, label = boxes[i]
        x1, y1, x2, y2, label = int(x1), int(y1), int(x2), int(y2), int(label)
        img_merge = cv2.rectangle(img_merge, (x1, y1), (x2, y2), (0, 255, 0), 2, 2)

    da_seg_mask = da_seg_mask * 255
    da_seg_mask = da_seg_mask.astype(np.uint8)
    da_seg_mask = cv2.resize(da_seg_mask, (width, height), interpolation=cv2.INTER_LINEAR)

    ll_seg_mask = ll_seg_mask * 255
    ll_seg_mask = ll_seg_mask.astype(np.uint8)
    ll_seg_mask = cv2.resize(ll_seg_mask, (width, height), interpolation=cv2.INTER_LINEAR)

    # 将可行驶区域和车道线叠加到原图像上
    imgbgr[da_seg_mask != 0] = imgbgr[da_seg_mask != 0] * 0.5 + [0, 255, 0] * 0.5
    imgbgr[ll_seg_mask != 0] = imgbgr[ll_seg_mask != 0] * 0.5 + [255, 0, 0] * 0.5

    # 叠加检测框
    for i in range(boxes.shape[0]):
        x1, y1, x2, y2, conf, label = boxes[i]
        x1, y1, x2, y2, label = int(x1), int(y1), int(x2), int(y2), int(label)
        imgbgr = cv2.rectangle(imgbgr, (x1, y1), (x2, y2), (0, 255, 0), 2, 2)

    return imgbgr

if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)
    #camera start
    try:
        cam = srcampy.Camera()
        ret = cam.open_cam(0, -1, 30, 1920, 1080)
        print("Camera open_cam return:%d" % ret)
    except Exception as e:
        print(f"Failed to initialize camera: {e}")
        exit(1)   

    #vps start
    try:
        vps1 = srcampy.Camera()
        vps2 = srcampy.Camera()
        ret1 = vps1.open_vps(1, 1, 1920, 1080, 672, 672)
        ret2 = vps2.open_vps(2, 1, 672, 672, 640, 640)
        if ret1 == 0 and ret2 == 0 :
            print("VPS channels opened successfully.")
            print("Camera vps ret return:%d " % ret1)
            print("Camera vps zen return:%d " % ret2)
            # print("Camera vps san return:%d " % san)
        else:
            print("vps Open Failed!")
            exit(1)
    except Exception as e:
        print(f"Failed to vps start: {e}")
        exit(1)
    #display start
    try:
        ##  初始化显示对象
        disp = srcampy.Display()
        ret = disp.display(0, 640, 640, 0, 1)
        print("Display display return:%d" % ret)
	##  将摄像头与现实对象绑定
        srcampy.bind(cam, disp)
        # disp.display(3, 640, 640)
    except Exception as e:
        print(f"Failed to Display display: {e}")
        exit(1)
    # 载入模型路径
    weight_path = './yolovp_640x640_nv12.bin'  
    is_stop = False
    a=1
    while not is_stop:  
        # 从摄像头获取图像数据
        img_nv12 = cam.get_img(0, 1920, 1080)  # 获取相机数据流
        if img_nv12 is not None:
            print("camera get image success ") 
            #print("img_nv12 get image is return:%d " % img_nv12)
        else:
            print("Failed to get image from camera:%d " % a)
            a = a+1
            if a == 20:
                break
            continue
        try:
            ## 第一路VPS处理，1080转672
            #send image data to vps  设置图片数据向vps
            ret = vps1.set_img(img_nv12)
            if ret != 0:
                print ("Failed to set image to first VPS.")
                continue        
        except Exception as e:
            print(f"set_img1 failed processing: {e}")
        
        img_vps1 = vps1.get_img(2, 672, 672)
        if img_vps1 is not None:
            print("img_vps1 get image success ")
        else:
            print("img_vps1 get image Failed")
            continue
        ## 第二路 VPS处理， 672转640
        try:
            ret = vps2.set_img(img_vps1)
            if ret != 0:
                print ("Failed to set image to second VPS.")
                continue       
        except Exception as e:
            print(f"img_vps1 failed processing: {e}")
        try:
            ret = vps2.set_img(img_vps1)
            if ret != 0:
                print ("Failed to set image to second VPS.")
                continue       
        except Exception as e:
            print(f"img_vps1 failed processing: {e}")

        try:
            img_vps2 = vps2.get_img(2, 640, 640)
            if img_vps2 is not None:
                print("img_vps2 get image success ")
            else:
                print("img_vps2 get image failed ")
                continue
            # 检查图像数据大小
            expected_size = 640 * 640 * 1.5  # NV12 格式的期望大小
            actual_size = len(img_vps2)
            if actual_size != expected_size:
                print(f"Invalid frame size: expected {expected_size}, got {actual_size}")
                continue
            # 在尝试设置显示之前，打印图像数据的地址和大小
            print(f"Image address: {id(img_vps2)}, Size: {actual_size}")
        except Exception as e:
            print(f"img_vps2 failed processing: {e}")

        # try:
        #     # 调用YOLOP进行处理
            # img_det = infer_yolop_camera(weight_path, img_vps2)  
        # except Exception as e:
        #     print(f"Error during is infer_yolop_camera: {e}")
        #     continue
        try:
            # # 将处理后的图像转换为 NV12 格式
            # img_nv12_det = bgr2nv12_opencv(img_det)
            # 在尝试设置显示之前，打印图像数据的地址和大小
            # 由于需要 NV12 格式，直接使用 VPS 输出，跳过转换步骤
            img_nv12_for_display = img_vps2
            print(f"Image address: {id(img_nv12_for_display)}, Size: {len(img_nv12_for_display)}")
        except Exception as e:
            print(f"Error during is bgr2nv12_opencv: {e}")
            continue    
        try: 
            ret = disp.set_img(img_nv12_for_display)
            if ret != 0:
                print("Display set image failed, addr:(nil) size:-1")
                continue
            print("Display set_img return:%d" % ret)
           
        except Exception as e:
            print(f"Error during processing: {e}")
            continue
        if cv2.waitKey(1) & 0xFF == ord('q'):
                is_stop = True
    disp.close()
    vps2.close_cam()
    cam.close_cam()
    print("test_cam_vps_display done!!!")
    cv2.destroyAllWindows()



