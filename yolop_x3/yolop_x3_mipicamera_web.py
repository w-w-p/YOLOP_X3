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
import time
import google.protobuf
import asyncio
import websockets
import x3_pb2
import subprocess
# Camera API libs
from lib.core.general import non_max_suppression
from hobot_vio import libsrcampy as srcampy
from hobot_dnn import pyeasy_dnn as dnn
import threading



fps = 30

image_counter = None

output_tensors = None

fcos_postprocess_info = None

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
    
def limit_display_cord(coor):
    coor[0] = max(min(1920, coor[0]), 0)
    # min coor is set to 2 not 0, leaving room for string display
    coor[1] = max(min(1080, coor[1]), 2)
    coor[2] = max(min(1920, coor[2]), 0)
    coor[3] = max(min(1080, coor[3]), 0)
    return coor
    
def infer_yolop_camera(weight,image_nv12):
    model_path = weight
    model = dnn.load(model_path)
    print(f"Load {model_path} done!")
    try:
        imgbgr = nv12_to_bgr(image_nv12, 640, 640)
    except Exception as e:
        print(f"Failed to nv12_to_bgr : {e}")
        exit(1) 
    if image_nv12 is not None:
        #save file
        fo = open("output.img","wb")
        fo.write(image_nv12)
        fo.close
        print("camera save img file success!")
    else:
        print("camera save img file failed!")
    
    img_input = np.frombuffer(image_nv12, dtype=np.uint8)
    # 代码使用模型对输入图像进行推理  取模型的输出
    preds = model[0].forward(img_input)

    strides = [8, 16, 32, 64, 128]
    for i in range(len(strides)):
        # 根据量化类型设置系统内存
        if (output_tensors[i].properties.quantiType == 0):
            output_tensors[i].sysMem[0].virAddr = ctypes.cast(preds[i].buffer.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), ctypes.c_void_p)
            output_tensors[i + 5].sysMem[0].virAddr = ctypes.cast(preds[i + 5].buffer.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), ctypes.c_void_p)
            output_tensors[i + 10].sysMem[0].virAddr = ctypes.cast(preds[i + 10].buffer.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), ctypes.c_void_p)
        else:  
            output_tensors[i].sysMem[0].virAddr = ctypes.cast(preds[i].buffer.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)), ctypes.c_void_p)
            output_tensors[i + 5].sysMem[0].virAddr = ctypes.cast(preds[i + 5].buffer.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)), ctypes.c_void_p)
            output_tensors[i + 10].sysMem[0].virAddr = ctypes.cast(preds[i + 10].buffer.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)), ctypes.c_void_p)
        # 调用后处理库进行处理
        libpostprocess.FcosdoProcess(output_tensors[i], output_tensors[i + 5], output_tensors[i + 10], fcos_postprocess_info, i)
    # 获取后处理结果
    result_str = get_Postprocess_result(ctypes.pointer(fcos_postprocess_info))  
    # 将结果字符串解码为UTF-8格式
    result_str = result_str.decode('utf-8') 
    # 记录后处理结束时间 
    t2 = time.time()
    print("FcosdoProcess time is :", (t2 - t1))
    # print(result_str)

    # draw result
    # 解析JSON字符串  
    data = json.loads(result_str[14:])  

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
        # return imgbgr
        return imgbgr, None
    # 将检测框的坐标缩放回原始图像尺寸
    # scale coords to original size.
    #boxes[:, 0] -= dw
    #boxes[:, 1] -= dh
    #boxes[:, 2] -= dw
    #boxes[:, 3] -= dh
    #boxes[:, :4] /= r

    print(f"detect {boxes.shape[0]} bounding boxes.")              ######新添加
    # data = []
    # for i in range(0, len(boxes)):
    #     result = {}
    #     result['id'] = 123
    #     result['score'] = 98.5
    #     result['bbox'] = boxes[i]
    #     result['name'] = ''
    #     data.append(result)

    return imgbgr, data

## 将检测结果数据序列化为 protobuf 格式
# FrameMessage：protobuf 中的消息模板  data: 包含检测结果的列表
def serialize(FrameMessage, data):
    if data is not None:
        for result in data: 
            # get class name
            # 创建一个新的 Target 对象，用于存储单个目标的相关信息
            Target = x3_pb2.Target()
            bbox = result['bbox']  # 矩形框位置信息  
            score = result['score']  # 获取目标的得分  
            id = int(result['id'])  # 获取目标的类别id  
            name = result['name']  # 获取目标的类别名称 
            
            # print(f"bbox: {bbox}, score: {score}, id: {id}, name: {name}")
            # 限制显示坐标在有效范围内
            bbox = limit_display_cord(bbox) ## 调用函数
            Target.type_ = classes[id] ## 设置目标的类别名称
            Box = x3_pb2.Box()      # 创建一个新的 Box 对象，用于存储检测框的坐标和得分。
            Box.type_ = classes[id] # 设置检测框的类别名称
            Box.score_ = float(score) # 设置检测框的得分

            # 设置检测框的左上角和右下角坐标
            Box.top_left_.x_ = int(bbox[0])
            Box.top_left_.y_ = int(bbox[1])
            Box.bottom_right_.x_ = int(bbox[2])
            Box.bottom_right_.y_ = int(bbox[3])
            # 将 Box 对象添加到 Target 对象中
            Target.boxes_.append(Box)
            FrameMessage.smart_msg_.targets_.append(Target)## 将 Target 对象添加到 protobuf 消息中
    # 将 protobuf 消息序列化为字符串。
    prot_buf = FrameMessage.SerializeToString() 
    return prot_buf  # 返回序列化后的字符串

##--- 以下这些步骤是进行图像处理和目标检测的准备工作 ---
# # 加载模型
# models = pyeasy_dnn.load('./fcos_512x512_nv12.bin')
# input_shape = (640, 640)
# cam = srcampy.Camera()
# cam.open_cam(0, -1, fps, [512,1920], [512,1080])  # 打开相机
# enc = srcampy.Encoder()
# enc.encode(0, 3, 1920, 1080)                # 解码
# classes = get_classes()
# # 打印输入 tensor 的属性
# print_properties(models[0].inputs[0].properties)
# print("--- model output properties ---")
# # 打印输出 tensor 的属性
# for output in models[0].outputs:
#     print_properties(output.properties)

# # 获取结构体信息
# fcos_postprocess_info = FcosPostProcessInfo_t()
# fcos_postprocess_info.height = 512   # 设置后处理信息，包括图像尺寸、阈值等
# fcos_postprocess_info.width = 512
# fcos_postprocess_info.ori_height = 1080
# fcos_postprocess_info.ori_width = 1920
# fcos_postprocess_info.score_threshold = 0.5 
# fcos_postprocess_info.nms_threshold = 0.6
# fcos_postprocess_info.nms_top_k = 500
# fcos_postprocess_info.is_pad_resize = 0
# # 初始化输出张量数组
# output_tensors = (hbDNNTensor_t * len(models[0].outputs))()
# # 配置输出张量的属性
# for i in range(len(models[0].outputs)):
#     output_tensors[i].properties.tensorLayout = get_TensorLayout(models[0].outputs[i].properties.layout)
#     # print(output_tensors[i].properties.tensorLayout)
#     if (len(models[0].outputs[i].properties.scale_data) == 0):
#         output_tensors[i].properties.quantiType = 0
#     else:
#         output_tensors[i].properties.quantiType = 2   
        
#         scale_data_tmp = models[0].outputs[i].properties.scale_data.reshape(1, 1, 1, models[0].outputs[i].properties.shape[3])  
#         output_tensors[i].properties.scale.scaleData = scale_data_tmp.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
#     for j in range(len(models[0].outputs[i].properties.shape)):
#         output_tensors[i].properties.validShape.dimensionSize[j] = models[0].outputs[i].properties.shape[j]
#         output_tensors[i].properties.alignedShape.dimensionSize[j] = models[0].outputs[i].properties.shape[j]

## 异步函数 用于提供webSocket服务
## 启动webSocket 服务，发送图像数据
async def web_service(websocket, path):
    
    weight_path = './yolovp_640x640_nv12.bin'  
    a=1    
    while True:
        # 创建一个protobuf消息对象
        FrameMessage = x3_pb2.FrameMessage()
        # 设置图像的高度和宽度
        FrameMessage.img_.height_ = 1080
        FrameMessage.img_.width_ = 1920
        # 设置图像类型为JPEG
        FrameMessage.img_.type_ = "JPEG"

        # # 从摄像头获取图像数据
        # img = cam.get_img(2, 640, 640)
        # # 将图像数据转换为NumPy数组
        # img = np.frombuffer(img, dtype=np.uint8)

        # 从摄像头获取图像数据
        img_nv12 = cam.get_img(2, 1920, 1080)  # 获取相机数据流
        if img_nv12 is not None:
            print("camera get image success ") 
            #print("img_nv12 get image is return:%d " % img_nv12)
        else:
            print("Failed to get image from camera:%d " % a)
            a = a+1
            if a == 20:
                break
            continue
        if img_nv12 is not None:
            #save file
            fo = open("output1.img","wb")
            fo.write(img_nv12)
            fo.close()
            print("camera save img_nv12 file success!")
        else:
            print("camera save img_nv12 file failed!")
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
        if img_vps1 is not None:
            #save file
            imgbgr = nv12_to_bgr(img_vps1, 672, 672)
            cv2.imwrite( "output_vps1.png",imgbgr)
            print("camera save output_vps1 file success!")
        else:
            print("camera save output_vps1 file failed!")
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

        if img_vps2 is not None:
            #save file
            imgbgr = nv12_to_bgr(img_vps2, 640, 640)
            cv2.imwrite( "img_vps2.png",imgbgr)
            print("camera save img_vps2 file success!")
        else:
            print("camera save img_vps2 file failed!")
        try:
            # 记录模型推理开始时间
            t0 = time.time()
            # 调用YOLOP进行处理
            img_bgr, data = infer_yolop_camera(weight_path, img_vps2)   ##  推理后的图像 
            t1 = time.time()
            print("forward time is :", (t1 - t0))
        except Exception as e:
            print(f"Error during is infer_yolop_camera: {e}")
            continue
        
        # 将图像转换为JPEG格式
        # _, buffer = cv2.imencode('.jpg', data)
        # # 创建FrameMessage对象并序列化
        # FrameMessage = x3_pb2.FrameMessage()
        # FrameMessage.img_.height_ = data.shape[0]
        # FrameMessage.img_.width_ = data.shape[1]
        # FrameMessage.img_.type_ = "JPEG"
        # FrameMessage.img_.buf_ = buffer.tobytes()

        # 获取原始图像数据  在原图基础上加上模型的框
        origin_image = img_nv12
        # 将原始图像数据编码为JPEG格式
        enc.encode_file(origin_image)
        # 将编码后的图像数据赋值给protobuf消息对象
        FrameMessage.img_.buf_ = enc.get_img()
        FrameMessage.smart_msg_.timestamp_ = int(time.time())
        # 从队列中获取图像
        prot_buf = serialize(FrameMessage, data)  ## 调用serialize函数 将检测结果数据序列化为protobuf格式
        # 通过WebSocket发送序列化后的数据
        await websocket.send(prot_buf)
    
        if cv2.waitKey(1) & 0xFF == ord('q'):
                is_stop = True
    disp.close()
    vps2.close_cam()
    cam.close_cam()
    print("web_service done!!!")
    cv2.destroyAllWindows()

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
    try:
        enc = srcampy.Encoder()
        enc.encode(0, 3, 1920, 1080)     # 解码
        print("Camera Encoder return:%d" % ret)
    except Exception as e:
        print(f"Failed to initialize Encoder: {e}")
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
    # #display start
    # try:
    #     ##  初始化显示对象
        disp = srcampy.Display()
        ret = disp.display(0, 640, 640, 0, 1)
        print("Display display return:%d" % ret)
	##  将摄像头与现实对象绑定
        srcampy.bind(cam, disp)
    #     # disp.display(3, 640, 640)
    # except Exception as e:
    #     print(f"Failed to Display display: {e}")
    #     exit(1)
    # 载入模型路径

    # 初始化WebSocket服务器
    start_server = websockets.serve(web_service, "0.0.0.0", 8080)
    asyncio.get_event_loop().run_until_complete(start_server)
    print("WebSocket server started at ws://0.0.0.0:8080")

    # try:
    #     # 主循环，用于处理图像并发送至Web端（省略，根据需要实现）
    #     # 这可能包括从相机获取图像、使用YOLOP模型进行检测等
    #     while True:
    #         pass  # 替换为实际的图像处理循环
    # except KeyboardInterrupt:
    #     # 处理退出信号
    #     print("Shutting down WebSocket server")
    # finally:
    #     # 清理资源（省略，调用原有的清理代码）
    #     pass

    # try: 
    #     ret = disp.set_img(img_nv12_for_display)
    #     if ret != 0:
    #         print("Display set image failed, addr:(nil) size:-1")
    #         continue
    #     print("Display set_img return:%d" % ret)
        
    # except Exception as e:
    #     print(f"Error during processing: {e}")
    #     continue
    asyncio.get_event_loop().run_forever()
