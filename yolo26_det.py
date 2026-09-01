"""
An example that uses TensorRT's Python api to make inferences.
"""
import ctypes
import os
import shutil
import argparse
import time
import cv2
import numpy as np
import pycuda.driver as cuda
import tensorrt as trt
import torch
torch.backends.cudnn.allow_tf32 = False
torch.backends.cuda.matmul.allow_tf32 = False

def get_tensor_np(engine, host_buf, tensor_name):
    shape = engine.get_tensor_shape(tensor_name)
    dtype = trt.nptype(engine.get_tensor_dtype(tensor_name))
    arr = np.array(host_buf, dtype=dtype).reshape(shape)
    return arr

def print_tensor_info(name:str, arr:np.ndarray):
    print(f"\n==== {name} ====")
    print(f"shape: {arr.shape}")
    print(f"max: {arr.max():.6f}, min: {arr.min():.6f}")
    print(f"mean: {arr.mean():.6f}")
    print("first 5 elements flat:", arr.flatten()[:5])
    arr.tofile("trt_"+ name +'.bin')


CLASS_NUM = 80
MAX_DET = 300
CONF_THREADS = 0.6
coco_names = {
    0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane',
    5: 'bus', 6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light',
    10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', 13: 'bench',
    14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow',
    20: 'elephant', 21: 'bear', 22: 'zebra', 23: 'giraffe', 24: 'backpack',
    25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase', 29: 'frisbee',
    30: 'skis', 31: 'snowboard', 32: 'sports ball', 33: 'kite', 34: 'baseball bat',
    35: 'baseball glove', 36: 'skateboard', 37: 'surfboard', 38: 'tennis racket',
    39: 'bottle', 40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife',
    44: 'spoon', 45: 'bowl', 46: 'banana', 47: 'apple', 48: 'sandwich',
    49: 'orange', 50: 'broccoli', 51: 'carrot', 52: 'hot dog', 53: 'pizza',
    54: 'donut', 55: 'cake', 56: 'chair', 57: 'couch', 58: 'potted plant',
    59: 'bed', 60: 'dining table', 61: 'toilet', 62: 'tv', 63: 'laptop',
    64: 'mouse', 65: 'remote', 66: 'keyboard', 67: 'cell phone', 68: 'microwave',
    69: 'oven', 70: 'toaster', 71: 'sink', 72: 'refrigerator', 73: 'book',
    74: 'clock', 75: 'vase', 76: 'scissors', 77: 'teddy bear', 78: 'hair drier',
    79: 'toothbrush'
}

trt_name = ["boxes","scores"]
def scale_boxes(img1_shape, boxes, img0_shape):
    gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
    pad_x = round((img1_shape[1] - round(img0_shape[1] * gain)) / 2 - 0.1)
    pad_y = round((img1_shape[0] - round(img0_shape[0] * gain)) / 2 - 0.1)

    boxes[..., 0] -= pad_x  # x padding
    boxes[..., 1] -= pad_y  # y padding
    boxes[..., 2] -= pad_x  # x padding
    boxes[..., 3] -= pad_y  # y padding
    boxes[..., :4] /= gain

    # clip_boxes
    h, w = img0_shape[:2]  # supports both HWC or HW shapes
    boxes[..., 0].clamp_(0, w)  # x1
    boxes[..., 1].clamp_(0, h)  # y1
    boxes[..., 2].clamp_(0, w)  # x2
    boxes[..., 3].clamp_(0, h)  # y2

    return boxes

def draw_yolo26_box(img: np.ndarray, dets: np.ndarray, names: dict, conf_thresh=0.25):
    img_draw = img.copy()
    line_width = max(round(sum(img_draw.shape) / 2 * 0.003), 2)  # 自适应线宽
    font_scale = line_width / 3
    for row in dets:
        x1, y1, x2, y2, conf, cls_id = row
        print(row)
        if conf < conf_thresh:
            continue
        cls_id = int(cls_id)
        # 坐标转为int
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        label = f"{names[cls_id]} {conf:.2f}"

        # 画框
        cv2.rectangle(img_draw, (x1, y1), (x2, y2), (0, 255, 0), thickness=line_width)
        # 绘制标签背景+文字
        (tw, th), bl = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, fontScale=font_scale, thickness=line_width)
        cv2.rectangle(img_draw, (x1, y1 - th - bl), (x1 + tw, y1), (0, 255, 0), -1)
        cv2.putText(img_draw, label, (x1, y1 - bl), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255,255,255), line_width)
    return img_draw


class YoLo26TRT(object):
    def __init__(self, engine_file_path):
        cuda.init()
        device = cuda.Device(0)
        self.ctx = device.make_context()  # 自动push，现在上下文激活

        try:
            runtime = trt.Runtime(trt.Logger(trt.Logger.INFO))
            with open(engine_file_path, "rb") as f:
                engine = runtime.deserialize_cuda_engine(f.read())

            self.engine = engine
            self.context = engine.create_execution_context()

            self.input_name = "images"
            self.output_name = "output"
            inp_shape = engine.get_tensor_shape(self.input_name)
            self.batch_size = inp_shape[0]
            self.input_h = inp_shape[-2]
            self.input_w = inp_shape[-1]
            print(f"engine.input: {inp_shape}")

            out_shape = engine.get_tensor_shape(self.output_name)
            self.det_output_length = out_shape[1]
            self.det_output_point = out_shape[2]
            print(f"engine.ouput: {out_shape}")

            # 上下文处于active，可以分配内存
            size_inp = trt.volume(inp_shape)
            dtype_inp = trt.nptype(engine.get_tensor_dtype(self.input_name))
            self.host_input = cuda.pagelocked_empty(size_inp, dtype_inp)
            self.cuda_input = cuda.mem_alloc(self.host_input.nbytes)

            size_out = trt.volume(out_shape)
            dtype_out = trt.nptype(engine.get_tensor_dtype(self.output_name))
            self.host_output = cuda.pagelocked_empty(size_out, dtype_out)
            self.cuda_output = cuda.mem_alloc(self.host_output.nbytes)

            self.context.set_tensor_address(self.input_name, int(self.cuda_input))
            self.context.set_tensor_address(self.output_name, int(self.cuda_output))

            # self.host_buffers = dict()
            # self.cuda_buffers = dict()
            # for i in range(engine.num_io_tensors):
            #     tensor_name = engine.get_tensor_name(i)
            #     if tensor_name in trt_name:
            #         tensor_shape = engine.get_tensor_shape(tensor_name)
            #         io_mode = engine.get_tensor_mode(tensor_name)
            #
            #         elem_count = trt.volume(tensor_shape)
            #         dtype = trt.nptype(engine.get_tensor_dtype(tensor_name))
            #         host_mem = cuda.pagelocked_empty(elem_count, dtype)
            #         dev_mem = cuda.mem_alloc(host_mem.nbytes)
            #
            #         self.host_buffers[tensor_name] = host_mem
            #         self.cuda_buffers[tensor_name] = dev_mem
            #
            #         # 绑定显存地址
            #         self.context.set_tensor_address(tensor_name, int(dev_mem))
            #
            #         print(f"index {i}: name = '{tensor_name}', mode = {io_mode}, shape = {tensor_shape}")

            self.stream = cuda.Stream()
        finally:
            self.ctx.pop()

    def destroy(self):
        self.ctx.push()
        try:
            if hasattr(self, "cuda_input"):
                self.cuda_input.free()
            if hasattr(self, "cuda_output"):
                self.cuda_output.free()
            del self.context
            del self.engine
        finally:
            self.ctx.pop()
        del self.ctx

    def infer(self, image_raw, image_path=None, output=None):
        self.ctx.push()
        try:
            stream = self.stream
            context = self.context

            # preprocess
            input_image = self.preprocess_image(image_raw)
            np.copyto(self.host_input, input_image.ravel())

            # inference
            start = time.time()
            cuda.memcpy_htod_async(self.cuda_input, self.host_input, stream)
            status = context.execute_async_v3(stream.handle)
            if not status:
                raise RuntimeError("execute_async_v3 return false")
            cuda.memcpy_dtoh_async(self.host_output, self.cuda_output, stream)
            stream.synchronize()
            end = time.time()

            # for name in trt_name:
            #     cuda.memcpy_dtoh(self.host_buffers[name], self.cuda_buffers[name])
            #     print_tensor_info(name, get_tensor_np(self.engine, self.host_buffers[name], name))
            # print(self.host_output[:20])

            # postprocess
            out = self.host_output.reshape(1, self.det_output_length, self.det_output_point)
            out = torch.from_numpy(out)

            result = self.postprocess(out)                  # 1,8400,84 -> 1,300,6
            preds_list = self.non_max_suppression(result)   # 1,300,6 -> List[Tensor]

            # plot
            preds = preds_list[0]
            preds[:, :4] = scale_boxes(input_image.shape[2:], preds[:, :4], image_raw.shape)
            if preds.numel() and bool(image_path):
                img_draw = draw_yolo26_box(image_raw, preds, coco_names)
                cv2.imwrite(output+os.path.basename(image_path), img_draw)
            return end - start
        finally:
            self.ctx.pop()

    def get_raw_image(self, img_path):
        return cv2.imread(img_path)  # (H, W, 3)

    def get_raw_image_zeros(self):
        return np.zeros([self.input_h, self.input_w, 3], dtype=np.uint8)

    def preprocess(self, img: np.ndarray):
        """
        单帧预处理
        :param img: HWC BGR uint8 原图
        :return: 预处理后 HWC BGR uint8 letterbox图
        """
        h, w = img.shape[:2]
        new_shape = (640, 640)
        r = min(new_shape[0] / h, new_shape[1] / w)
        new_unpad = round(w * r), round(h * r)

        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
        dw /= 2
        dh /= 2
        top, bottom = round(dh - 0.1), round(dh + 0.1)
        left, right = round(dw - 0.1), round(dw + 0.1)

        # resize
        if (w, h) != new_unpad:
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)

        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114,) * 3)
        return img

    def preprocess_image(self, img: np.ndarray) -> torch.Tensor:
        """
        单帧完整预处理：letterbox -> BGR2RGB -> BHWC转BCHW -> float /255
        :param img: HWC BGR uint8
        :return: torch.Tensor [1,3,640,640] float32 0~1
        """
        # letterbox
        img = self.preprocess(img)
        im = np.expand_dims(img, axis=0)
        # BGR to RGB
        if im.shape[-1] == 3:
            im = im[..., ::-1]
        # BHWC -> BCHW
        im = im.transpose((0, 3, 1, 2))
        im = np.ascontiguousarray(im)
        im = torch.from_numpy(im)
        # im = im.half() if False else im.float()  # uint8 to fp16/32
        im = im.float()
        im /= 255.0
        return im

    def _grouped_topk(self, x: torch.Tensor, k: int, groups: int = 8) -> tuple[torch.Tensor, torch.Tensor]:
        n = x.shape[1]
        while groups > 1 and (n % groups or n // groups < k):
            groups //= 2
        if groups == 1:  # nothing to gain, e.g. a short axis or one that does not divide evenly
            return x.topk(k, dim=1)
        size = n // groups
        values, index = x.reshape(x.shape[0], groups, size).topk(k, dim=-1)
        values, winners = values.flatten(1).topk(k, dim=1)
        return values, winners // k * size + index.flatten(1).gather(1, winners)

    def get_topk_index(self, scores: torch.Tensor, max_det: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, anchors, nc = scores.shape  # i.e. shape(16,8400,80)
        k = min(max_det, anchors)
        groups = 1
        ori_index = self._grouped_topk(scores.max(dim=-1)[0], k, groups)[1].unsqueeze(-1)
        scores = scores.gather(dim=1, index=ori_index.expand(-1, -1, nc))
        scores, index = self._grouped_topk(scores.flatten(1), k, groups)
        idx =  ori_index.gather(dim=1, index=(index // nc).unsqueeze(-1))
        return scores[..., None], (index % nc)[..., None].float(), idx

    def postprocess(self, preds: torch.Tensor) -> torch.Tensor:
        boxes, scores = preds.split([4, CLASS_NUM], dim=-1)
        scores, conf, idx = self.get_topk_index(scores, MAX_DET)
        boxes = boxes.gather(dim=1, index=idx.expand(-1, -1, 4))
        return torch.cat([boxes, scores, conf], dim=-1)

    def non_max_suppression(self, prediction):
        output = []
        for pred in prediction:
            mask = pred[:, 4] > CONF_THREADS
            idx = mask.nonzero(as_tuple=False).view(-1)[:30]
            output.append(pred[idx])

        return output

def read_coco_classes(txt_path: str) -> list:
    categories = []
    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            _, cls_name = line.split(":", maxsplit=1)
            cls_name = cls_name.strip()
            categories.append(cls_name)
    return categories

def create_output(file_dir):
    if os.path.exists(file_dir):
        shutil.rmtree(file_dir)
    os.makedirs(file_dir)

def get_imgs_path(folder):
    img_list = []
    for name in os.listdir(folder):
        if name.lower().endswith((".jpg", ".png", ".jpeg")):
            img_list.append(os.path.join(folder, name))
    return img_list

def parse_args():
    parser = argparse.ArgumentParser(description="YOLO26 TensorRT Inference Demo")
    parser.add_argument("--engine", type=str, default="file/yolo26s.engine", help="Path of TensorRT engine file")
    parser.add_argument("--plugin", type=str, default="build/libtrt_plugins.so", help="Path of plugin library")
    parser.add_argument("--classes", type=str, default="file/coco.txt", help="Path of COCO class names file")
    parser.add_argument("--input", type=str, default="images/", help="Input images directory")
    parser.add_argument("--warmup", type=int, default=10, help="GPU warm-up iterations")
    parser.add_argument("--output", type=str, default="output/", help="Output directory")
    return parser.parse_args()


if __name__ == "__main__":
    # official
    from ultralytics import YOLO
    yolo_model = YOLO("./file/yolo26s.yaml").load("./file/yolo26s.pt")
    yolo_model.eval()
    yolo_model("./images/bus.png", conf=CONF_THREADS, save=True)
    print()

    # trt
    args = parse_args()
    ctypes.CDLL(args.plugin)
    categories = read_coco_classes(args.classes)
    create_output(args.output)

    yolo26_wrapper = YoLo26TRT(args.engine)
    images_path = get_imgs_path(args.input)

    try:
        print("warmup start")
        for i in range(args.warmup):
            use_time = yolo26_wrapper.infer(yolo26_wrapper.get_raw_image_zeros())
            print(f'warm_up -> {i}: time -> {use_time * 1000:.2f}ms')

        for i, image_path in enumerate(images_path):
            use_time = yolo26_wrapper.infer(yolo26_wrapper.get_raw_image(image_path),image_path, args.output)
            print(f'inference -> {image_path}: time -> {use_time * 1000:.2f}ms')
    finally:
        yolo26_wrapper.destroy()
        import gc
        gc.collect()

