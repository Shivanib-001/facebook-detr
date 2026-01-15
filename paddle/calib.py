# mask_rtdetr_int8_calibration_jetson.py

import os
import cv2
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit


############################################
# CONFIG (EDIT THESE)
############################################
ONNX_MODEL_PATH = "mask_rtdetr.onnx"
CALIB_IMAGE_DIR = "calib_images"
ENGINE_PATH = "mask_rtdetr_int8.engine"
CALIB_CACHE = "mask_rtdetr_int8.cache"

CALIB_BATCH_SIZE = 1
CALIB_MAX_SAMPLES = 200

OPT_H, OPT_W = 800, 800
MIN_H, MIN_W = 640, 640
MAX_H, MAX_W = 1024, 1024
############################################


TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE)

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def preprocess_image(img_path, target_hw=(OPT_H, OPT_W)):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    h, w = img.shape[:2]
    orig_size = np.array([h, w], dtype=np.int32)

    img = cv2.resize(img, target_hw)
    img = img.astype(np.float32) / 255.0
    img = (img - IMAGENET_MEAN) / IMAGENET_STD
    img = np.transpose(img, (2, 0, 1))  # CHW
    img = np.expand_dims(img, axis=0)   # B=1

    return img, np.expand_dims(orig_size, axis=0)


class CalibrationDataLoader:
    def __init__(self, image_dir, max_samples):
        self.images = [
            os.path.join(image_dir, f)
            for f in os.listdir(image_dir)
            if f.lower().endswith((".jpg", ".png"))
        ][:max_samples]
        self.idx = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.idx >= len(self.images):
            raise StopIteration

        img, size = preprocess_image(self.images[self.idx])
        self.idx += 1
        return img, size


class MaskRTDETRInt8Calibrator(trt.IInt8EntropyCalibrator2):
    def __init__(self, dataloader, cache_file):
        super().__init__()
        self.dataloader = dataloader
        self.cache_file = cache_file
        self.iterator = iter(dataloader)

        self.device_images = cuda.mem_alloc(
            CALIB_BATCH_SIZE * 3 * OPT_H * OPT_W * 4
        )
        self.device_sizes = cuda.mem_alloc(
            CALIB_BATCH_SIZE * 2 * 4
        )

    def get_batch_size(self):
        return CALIB_BATCH_SIZE

    def get_batch(self, names):
        try:
            images, sizes = next(self.iterator)
            cuda.memcpy_htod(self.device_images, images)
            cuda.memcpy_htod(self.device_sizes, sizes)
            return [int(self.device_images), int(self.device_sizes)]
        except StopIteration:
            return None

    def read_calibration_cache(self):
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "rb") as f:
                return f.read()
        return None

    def write_calibration_cache(self, cache):
        with open(self.cache_file, "wb") as f:
            f.write(cache)


def build_int8_engine():
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser = trt.OnnxParser(network, TRT_LOGGER)

    with open(ONNX_MODEL_PATH, "rb") as f:
        if not parser.parse(f.read()):
            print("❌ ONNX parse failed")
            for i in range(parser.num_errors):
                print(parser.get_error(i))
            return

    config = builder.create_builder_config()

    profile = builder.create_optimization_profile()
    profile.set_shape(
        "images",
        min=(1, 3, MIN_H, MIN_W),
        opt=(1, 3, OPT_H, OPT_W),
        max=(1, 3, MAX_H, MAX_W),
    )
    profile.set_shape(
        "orig_target_sizes",
        min=(1, 2),
        opt=(1, 2),
        max=(1, 2),
    )

    config.add_optimization_profile(profile)

    config.set_flag(trt.BuilderFlag.INT8)
    config.set_flag(trt.BuilderFlag.FP16)
    config.default_device_type = trt.DeviceType.GPU

    config.set_memory_pool_limit(
        trt.MemoryPoolType.WORKSPACE, 4 << 30
    )

    dataloader = CalibrationDataLoader(
        CALIB_IMAGE_DIR, CALIB_MAX_SAMPLES
    )
    calibrator = MaskRTDETRInt8Calibrator(
        dataloader, CALIB_CACHE
    )
    config.int8_calibrator = calibrator

    print("🚀 Building INT8 engine on Jetson Orin (this will take time)...")
    engine = builder.build_engine(network, config)

    if engine is None:
        print("❌ Engine build failed")
        return

    with open(ENGINE_PATH, "wb") as f:
        f.write(engine.serialize())

    print("✅ INT8 engine built successfully:", ENGINE_PATH)


if __name__ == "__main__":
    build_int8_engine()
