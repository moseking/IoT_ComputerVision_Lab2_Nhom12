"""YOLOv8 three-class FP32 ONNX inference, without Torch/Ultralytics at runtime."""
import ast
import hashlib
import json
import time
import cv2
import numpy as np
from .inference import CLASSES, MaskDetectionModel, InvalidImage


def agnostic_nms(boxes, scores, threshold, limit=300):
    order = np.argsort(-scores, kind="stable")
    area = np.maximum(0, boxes[:, 2]-boxes[:, 0]) * np.maximum(0, boxes[:, 3]-boxes[:, 1])
    keep = []
    while order.size and len(keep) < limit:
        i = int(order[0]); keep.append(i)
        rest = order[1:]
        if not rest.size: break
        left = np.maximum(boxes[i, :2], boxes[rest, :2])
        right = np.minimum(boxes[i, 2:], boxes[rest, 2:])
        inter = np.maximum(0, right-left).prod(axis=1)
        overlap = inter / (area[i] + area[rest] - inter + 1e-7)
        order = rest[overlap <= threshold]
    return keep


class OnnxMaskDetectionModel(MaskDetectionModel):
    def __init__(self, path, confidence=.35, iou=.45, image_size=512, expected_sha256=None, max_pixels=12000000):
        super().__init__(path, confidence, iou, image_size, "cpu", max_pixels)
        self.expected_sha256 = expected_sha256
        self.artifact_sha256 = None
        self._session = None
        self._warmed = False

    @property
    def ready(self):
        return self._session is not None and self._warmed

    def load_model(self):
        with self._lock:
            if self.ready: return
            if not self.path.is_file(): raise FileNotFoundError("Cloud ONNX model is missing")
            self.artifact_sha256 = hashlib.sha256(self.path.read_bytes()).hexdigest()
            if self.expected_sha256 and self.artifact_sha256 != self.expected_sha256:
                raise ValueError("Cloud model SHA-256 mismatch")
            import onnxruntime as ort
            options = ort.SessionOptions()
            options.intra_op_num_threads = 1
            options.inter_op_num_threads = 1
            options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            session = ort.InferenceSession(str(self.path), sess_options=options, providers=["CPUExecutionProvider"])
            metadata = session.get_modelmeta().custom_metadata_map
            mapping = (json.loads(metadata["mask_class_map_json"]) if "mask_class_map_json" in metadata
                       else ast.literal_eval(metadata.get("names", "{}")))
            if {int(k):v for k,v in mapping.items()} != CLASSES:
                raise ValueError("Cloud model classes mismatch")
            if metadata.get("task") != "detect" or metadata.get("end2end", "False").lower() == "true":
                raise ValueError("Expected YOLOv8 raw detection output")
            inputs, outputs = session.get_inputs(), session.get_outputs()
            if len(inputs) != 1 or len(outputs) != 1 or inputs[0].type != "tensor(float)" or inputs[0].shape[1] != 3:
                raise ValueError("Invalid cloud model input/output contract")
            self._input_name = inputs[0].name
            self._dynamic = all(not isinstance(d, int) for d in inputs[0].shape[2:])
            self._session = session
            started = time.perf_counter()
            self._infer(np.zeros((self.image_size,self.image_size,3),dtype=np.uint8))
            self.warmup_ms = round((time.perf_counter()-started)*1000,3)
            self._warmed = True

    def _infer(self, image):
        h,w = image.shape[:2]
        scale = min(self.image_size/h,self.image_size/w)
        new_w,new_h = round(w*scale),round(h*scale)
        dw,dh = self.image_size-new_w,self.image_size-new_h
        if self._dynamic: dw,dh = dw%32,dh%32
        left,right = round(dw/2-.1),round(dw/2+.1)
        top,bottom = round(dh/2-.1),round(dh/2+.1)
        resized = cv2.resize(image,(new_w,new_h),interpolation=cv2.INTER_LINEAR) if (new_w,new_h)!=(w,h) else image
        padded = cv2.copyMakeBorder(resized,top,bottom,left,right,cv2.BORDER_CONSTANT,value=(114,114,114))
        tensor = np.ascontiguousarray(padded[:,:,::-1].transpose(2,0,1)[None],dtype=np.float32)/255.0
        raw = self._session.run(None,{self._input_name:tensor})[0]
        if raw.ndim!=3 or raw.shape[0]!=1 or raw.shape[1]!=7:
            raise RuntimeError("Unexpected YOLOv8 cloud output shape")
        predictions = raw[0].T
        classes = np.argmax(predictions[:,4:],axis=1)
        scores = predictions[np.arange(len(predictions)),classes+4]
        selected = np.isfinite(predictions).all(axis=1) & (scores>self.confidence)
        rows,classes,scores = predictions[selected],classes[selected],scores[selected]
        boxes = np.empty((len(rows),4),dtype=np.float32)
        boxes[:,:2] = rows[:,:2]-rows[:,2:4]/2
        boxes[:,2:] = rows[:,:2]+rows[:,2:4]/2
        keep = agnostic_nms(boxes,scores,self.iou)
        # Match Ultralytics 8.4.174 scale_boxes: each rounded resize axis
        # has its own effective gain rather than the unrounded scalar ratio.
        gain=min(padded.shape[0]/h,padded.shape[1]/w)
        rounded_h,rounded_w=round(h*gain),round(w*gain)
        pad_x=round((padded.shape[1]-rounded_w)/2-.1)
        pad_y=round((padded.shape[0]-rounded_h)/2-.1)
        boxes[:,[0,2]] = np.clip((boxes[:,[0,2]]-pad_x)/(rounded_w/w),0,w)
        boxes[:,[1,3]] = np.clip((boxes[:,[1,3]]-pad_y)/(rounded_h/h),0,h)
        return [{"class_id":int(classes[i]),"class_name":CLASSES[int(classes[i])],
                 "confidence":round(float(scores[i]),6),
                 "bbox":dict(zip(("x1","y1","x2","y2"),(round(float(v),2) for v in boxes[i])))} for i in keep]

    def predict(self, image):
        if not isinstance(image,np.ndarray) or image.ndim!=3 or image.shape[2]!=3:
            raise InvalidImage("Expected a BGR image")
        if image.shape[0]*image.shape[1]>self.max_pixels: raise InvalidImage("Image exceeds pixel limit")
        queued=time.perf_counter()
        with self._lock:
            if not self.ready: raise RuntimeError("Cloud model is not ready")
            start=time.perf_counter();detections=self._infer(image);elapsed=(time.perf_counter()-start)*1000
        return {"detections":detections,"inference_ms":round(elapsed,3),"queue_ms":round((start-queued)*1000,3),
                "inference_definition":"cloud CPU wall_time_preprocess_ONNX_forward_agnostic_NMS; excludes queue/decode/annotation"}
