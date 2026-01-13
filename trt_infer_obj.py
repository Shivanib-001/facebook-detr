# infer_detr_trt_video.py
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision.ops import nms
import pycuda.driver as cuda
import pycuda.autoinit
import tensorrt as trt
import argparse

# --- logger ---
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

transform = T.Compose([
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225])
])

COCO_CLASSES = ['NA','Trunk', 'NA']

def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('--video_path', type=str)
    parser.add_argument('--model_path', type=str)
    parser.add_argument('--display', default=False, type=bool, help='set display yo true to enable visualisation')
    parser.add_argument('--score', default=0.1, type=float, help='set score threshold')
    return parser
   
'''def preprocess(img):

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_t = transform(img_rgb).unsqueeze(0).numpy()

    return np.ascontiguousarray(img_t, dtype=np.float32)'''

def preprocess(img):
    img = torch.from_numpy(img).cuda(non_blocking=True)
    img = img.permute(2, 0, 1).unsqueeze(0) / 255.0

    mean = torch.tensor([0.485, 0.456, 0.406], device="cuda").view(1,3,1,1)
    std  = torch.tensor([0.229, 0.224, 0.225], device="cuda").view(1,3,1,1)

    img = (img - mean) / std
    img = img[:, [2,1,0]]
    return img.contiguous()  # only for TRT input

def rescale_bboxes(boxes,w,h):

    cx, cy, bw, bh = boxes.unbind(-1)
    x1 = (cx - bw / 2) * w
    y1 = (cy - bh / 2) * h
    x2 = (cx + bw / 2) * w
    y2 = (cy + bh / 2) * h
    return torch.stack([x1, y1, x2, y2], dim=-1)
   

class TRTInference:
    def __init__(self, engine_path):

        cuda.init()
        self.device = cuda.Device(0)
        
        self.cfx = self.device.make_context()  
        self.engine = self.load_engine(engine_path)
        self.context = self.engine.create_execution_context()
        self.stream = cuda.Stream()
        self.device_buffers = {}
        self.host_outputs = {}
        self.input_shape = (1, 3, 720, 1280)
        self.allocate_buffers()
        for name in self.device_buffers:
            self.context.set_tensor_address(name, int(self.device_buffers[name]))
        self.cfx.pop()

    def load_engine(self, engine_path):
        with open(engine_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            return runtime.deserialize_cuda_engine(f.read())
    
    def allocate_buffers(self):
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            dtype = trt.nptype(self.engine.get_tensor_dtype(name))
            shape = self.context.get_tensor_shape(name)
            size = trt.volume(shape)
            device_mem = cuda.mem_alloc(size * np.dtype(dtype).itemsize)
            self.device_buffers[name] = device_mem
           
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT:
                self.host_outputs[name] = np.empty(size, dtype=dtype)
               


    def infer(self, input_data):

        self.cfx.push()

        for name in self.device_buffers:
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.context.set_input_shape(name, self.input_shape)
                cuda.memcpy_dtod_async(int(self.device_buffers[name]),int(input_data.data_ptr()),input_data.numel() * input_data.element_size(),self.stream)
        
        self.context.execute_async_v3(self.stream.handle)
        
        for name, host_out in self.host_outputs.items():
            cuda.memcpy_dtoh_async(host_out, self.device_buffers[name], self.stream)
            
        self.stream.synchronize()
        
        outputs={}
        for name in sorted(self.host_outputs.keys()):
            shape = self.context.get_tensor_shape(name)
            outputs[name]=(self.host_outputs[name].reshape(shape))

        self.cfx.pop()
        return outputs  

import time

# --- Video inference ---
def run_video(video_path, engine_path, display, score_thresh, nms_thresh=0.5):
    trt_infer = TRTInference(engine_path)
    cap = cv2.VideoCapture(video_path)

    start = time.time()
    count = 0
    device = torch.device("cuda")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        orig_img = frame.copy()
        count += 1
        h, w = frame.shape[:2]

        img_t = preprocess(frame)

        outputs = trt_infer.infer(img_t)

        # unpack
        pred_logits = outputs["pred_logits"]
        pred_boxes  = outputs["pred_boxes"]

        # ---------------- GPU POSTPROCESS ----------------
        logits = torch.from_numpy(pred_logits[0]).to(device)
        boxes  = torch.from_numpy(pred_boxes[0]).to(device)

        probs = torch.softmax(logits, dim=-1)
        scores, labels = probs.max(dim=-1)

        keep = scores > score_thresh
        if keep.any():
            boxes  = boxes[keep]
            scores = scores[keep]
            labels = labels[keep]

            boxes=rescale_bboxes(boxes,w,h)
            keep_idx = nms(boxes, scores, nms_thresh)
            
            #--------let be back to cpu ---------------------
            boxes  = boxes[keep_idx].int().cpu().numpy()
            scores = scores[keep_idx].cpu().numpy()
            labels = labels[keep_idx].cpu().numpy()

            for bbox, score, cls_id in zip(boxes, scores, labels):
                cls_name = COCO_CLASSES[cls_id] if cls_id < len(COCO_CLASSES) else str(cls_id)
                if cls_name == "Trunk":
                    cv2.rectangle( orig_img,(bbox[0], bbox[1]),(bbox[2], bbox[3]),(0, 255, 0),  2)
                    cv2.putText(orig_img,f"{cls_name}:{score:.2f}",(bbox[0], max(bbox[1] - 10, 0)),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0, 255, 0),2 )

        if display:
            cv2.imshow("DETR TRT Video Inference", orig_img)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    end = time.time()
    fps = count / (end - start)
    print("average fps :", fps)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_video("/home/rnil/Documents/model/yolact-all/test_images/test_video1.mp4", "/home/rnil/Documents/model/transformer/detr_/detr1/test_box.engine",True,0.1)

    
    '''parser = argparse.ArgumentParser('DETR ONNX box eval script', parents=[get_args_parser()])
    args = parser.parse_args()
    if (args.score>0 and args.score<1):
     
        run_video(args.video_path, args.model_path, args.display, args.score)
    else:
        raise ValueError("score threshold must be between 0 and 1")'''
