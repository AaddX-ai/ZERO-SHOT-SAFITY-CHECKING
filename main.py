import onnxruntime as ort
import numpy as np
import cv2
from transformers import CLIPSegProcessor
import time
import tkinter as tk
from tkinter import simpledialog

# 加载ONNX模型
# 创建会话选项
session_options = ort.SessionOptions()
# 启用 FP16 推理
session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
session_options.enable_profiling = True

ort_session = ort.InferenceSession("clipseg_model.onnx",session_options)
processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")

def process_image(image, prompt="a large white block area", threshold=0.8):
 # 处理图像和文本
 inputs = processor(text=prompt, images=image, return_tensors="np")

 inputs = {
     'input_ids': inputs['input_ids'],
     'pixel_values': inputs['pixel_values'],
     'attention_mask': inputs['attention_mask']
 }

 # 根据模型的输入大小填充输入
 input_ids_padded = np.zeros((1, 77), dtype=np.int32)
 attention_mask_padded = np.zeros((1, 77), dtype=np.int32)
 input_ids_padded[:, :inputs['input_ids'].shape[1]] = inputs['input_ids']
 attention_mask_padded[:, :inputs['attention_mask'].shape[1]] = inputs['attention_mask']

 inputs['input_ids'] = input_ids_padded
 inputs['attention_mask'] = attention_mask_padded

 start_time = time.time()
 logits = ort_session.run(None, inputs)[0]  # 假设logits是第一个输出
 inference_time = time.time() - start_time
 print(f"Inference time: {inference_time:.4f} seconds")

 # 使用sigmoid处理logits以获取掩码概率
 prob = 1 / (1 + np.exp(-logits))  # Sigmoid函数

 # 将概率转换为图像
 mask = (prob * 255).astype(np.uint8)  # 转换为uint8格式
 mask = cv2.resize(mask, (image.shape[1], image.shape[0]))

 # 归一化和阈值处理掩码
 mask_min = mask.min()
 mask_max = mask.max()
 mask = (mask - mask_min) / (mask_max - mask_min)
 bmask = mask > threshold
 mask[bmask] = 1
 mask[~bmask] = 0

 # 转换回uint8格式以便保存
 mask_image = (mask * 255).astype(np.uint8)

 contours, hierarchy = cv2.findContours(bmask.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
 bounding_boxes = []
 for contour in contours:
     x, y, w, h = cv2.boundingRect(contour)
     bounding_boxes.append((x, y, w, h))

 return bounding_boxes, mask_image

def check_overlap(bbox1, bbox2):
 # 检查两个边界框是否重叠
 x1, y1, w1, h1 = bbox1
 x2, y2, w2, h2 = bbox2
 return not (x1 > x2 + w2 or x2 > x1 + w1 or y1 > y2 + h2 or y2 > y1 + h1)




# 示例使用
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# 缓存最近4帧的边界框
bbox_history = []
prompt = "hand"  # 默认提示

frame_index=6
while True:
 ret, frame = cap.read()
 if not ret:
     continue
 h, w, _ = frame.shape
 start_x = (w - 720) // 2
 start_y = (h - 720) // 2
 image = frame[start_y:start_y + 720, start_x:start_x + 720]
 image = cv2.resize(image, (352, 352))
   
 bounding_boxes, mask_image = process_image(image, prompt=prompt, threshold=0.8)

 # 更新边界框历史记录
 bbox_history.append(bounding_boxes)
 if len(bbox_history) > frame_index:
     bbox_history.pop(0)  # 只保留最近4帧

 # 检查最近4帧中的重叠边界框
 if len(bbox_history) == frame_index:
     overlapping_bboxes = []
    
     # 遍历第一个边界框
     for i in range(len(bbox_history[0])):
         current_bbox = bbox_history[0][i]
         overlaps = 0
        
         # 与其他三帧的边界框进行比较
         for j in range(1, frame_index):
             if j < len(bbox_history) and len(bbox_history[j]) > 0:
                 for bbox in bbox_history[j]:
                     if check_overlap(current_bbox, bbox):
                         overlaps += 1
                         break  # 对于这个边界框不需要进一步检查

         # 如果所有帧都有重叠，添加到重叠边界框列表
         if overlaps == frame_index-1:  # 检查与3帧的重叠
             overlapping_bboxes.append(current_bbox)

     # 在当前图像上绘制重叠的边界框
     for bbox in set(overlapping_bboxes):
         x, y, w, h = bbox
         cv2.rectangle(image, (x, y), (x + w, y + h), color=(0, 255, 255), thickness=1)  # 黄色边框

 cv2.imshow('src', image)
 # cv2.imshow('result', mask_image)
 if cv2.waitKey(30) & 0xFF == ord("q"):
     break

cap.release()
cv2.destroyAllWindows()import onnxruntime as ort
import numpy as np
import cv2
from transformers import CLIPSegProcessor
import time
import tkinter as tk
from tkinter import simpledialog

# 加载ONNX模型
# 创建会话选项
session_options = ort.SessionOptions()
# 启用 FP16 推理
session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
session_options.enable_profiling = True

ort_session = ort.InferenceSession("clipseg_model.onnx",session_options)
processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")

def process_image(image, prompt="a large white block area", threshold=0.8):
 # 处理图像和文本
 inputs = processor(text=prompt, images=image, return_tensors="np")

 inputs = {
     'input_ids': inputs['input_ids'],
     'pixel_values': inputs['pixel_values'],
     'attention_mask': inputs['attention_mask']
 }

 # 根据模型的输入大小填充输入
 input_ids_padded = np.zeros((1, 77), dtype=np.int32)
 attention_mask_padded = np.zeros((1, 77), dtype=np.int32)
 input_ids_padded[:, :inputs['input_ids'].shape[1]] = inputs['input_ids']
 attention_mask_padded[:, :inputs['attention_mask'].shape[1]] = inputs['attention_mask']

 inputs['input_ids'] = input_ids_padded
 inputs['attention_mask'] = attention_mask_padded

 start_time = time.time()
 logits = ort_session.run(None, inputs)[0]  # 假设logits是第一个输出
 inference_time = time.time() - start_time
 print(f"Inference time: {inference_time:.4f} seconds")

 # 使用sigmoid处理logits以获取掩码概率
 prob = 1 / (1 + np.exp(-logits))  # Sigmoid函数

 # 将概率转换为图像
 mask = (prob * 255).astype(np.uint8)  # 转换为uint8格式
 mask = cv2.resize(mask, (image.shape[1], image.shape[0]))

 # 归一化和阈值处理掩码
 mask_min = mask.min()
 mask_max = mask.max()
 mask = (mask - mask_min) / (mask_max - mask_min)
 bmask = mask > threshold
 mask[bmask] = 1
 mask[~bmask] = 0

 # 转换回uint8格式以便保存
 mask_image = (mask * 255).astype(np.uint8)

 contours, hierarchy = cv2.findContours(bmask.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
 bounding_boxes = []
 for contour in contours:
     x, y, w, h = cv2.boundingRect(contour)
     bounding_boxes.append((x, y, w, h))

 return bounding_boxes, mask_image

def check_overlap(bbox1, bbox2):
 # 检查两个边界框是否重叠
 x1, y1, w1, h1 = bbox1
 x2, y2, w2, h2 = bbox2
 return not (x1 > x2 + w2 or x2 > x1 + w1 or y1 > y2 + h2 or y2 > y1 + h1)




# 示例使用
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# 缓存最近4帧的边界框
bbox_history = []
prompt = "hand"  # 默认提示

frame_index=6
while True:
 ret, frame = cap.read()
 if not ret:
     continue
 h, w, _ = frame.shape
 start_x = (w - 720) // 2
 start_y = (h - 720) // 2
 image = frame[start_y:start_y + 720, start_x:start_x + 720]
 image = cv2.resize(image, (352, 352))
   
 bounding_boxes, mask_image = process_image(image, prompt=prompt, threshold=0.8)

 # 更新边界框历史记录
 bbox_history.append(bounding_boxes)
 if len(bbox_history) > frame_index:
     bbox_history.pop(0)  # 只保留最近4帧

 # 检查最近4帧中的重叠边界框
 if len(bbox_history) == frame_index:
     overlapping_bboxes = []
    
     # 遍历第一个边界框
     for i in range(len(bbox_history[0])):
         current_bbox = bbox_history[0][i]
         overlaps = 0
        
         # 与其他三帧的边界框进行比较
         for j in range(1, frame_index):
             if j < len(bbox_history) and len(bbox_history[j]) > 0:
                 for bbox in bbox_history[j]:
                     if check_overlap(current_bbox, bbox):
                         overlaps += 1
                         break  # 对于这个边界框不需要进一步检查

         # 如果所有帧都有重叠，添加到重叠边界框列表
         if overlaps == frame_index-1:  # 检查与3帧的重叠
             overlapping_bboxes.append(current_bbox)

     # 在当前图像上绘制重叠的边界框
     for bbox in set(overlapping_bboxes):
         x, y, w, h = bbox
         cv2.rectangle(image, (x, y), (x + w, y + h), color=(0, 255, 255), thickness=1)  # 黄色边框

 cv2.imshow('src', image)
 # cv2.imshow('result', mask_image)
 if cv2.waitKey(30) & 0xFF == ord("q"):
     break

cap.release()
cv2.destroyAllWindows()
