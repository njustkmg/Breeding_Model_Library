import os
import time
import torch
import torchvision
import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda
import tensorrt as trt
from PIL import Image

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE)


class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        """
        host_mem: cpu memory
        device_mem: gpu memory
        """
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host)+"\nDevice:\n"+str(self.device)

    def __repr__(self):
        return self.__str__()


def build_engine(onnx_file_path, engine_file_path, max_batch_size=1, fp16_mode=False, save_engine=False):
    """
    Args:
      max_batch_size: 预先指定大小好分配显存
      fp16_mode:      是否采用FP16
      save_engine:    是否保存引擎
    return:
      ICudaEngine
    """
    if os.path.exists(engine_file_path):
        print("Reading engine from file: {}".format(engine_file_path))
        with open(engine_file_path, 'rb') as f, trt.Runtime(TRT_LOGGER) as runtime:
            return runtime.deserialize_cuda_engine(f.read())  # 反序列化

    # 如果是动态输入，需要显式指定EXPLICIT_BATCH
    EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    # builder创建计算图 INetworkDefinition
    with trt.Builder(TRT_LOGGER) as builder, \
        builder.create_network(EXPLICIT_BATCH) as network, \
        builder.create_builder_config() as config,   trt.OnnxParser(network, TRT_LOGGER) as parser: 
        builder.max_workspace_size = 1 << 60           # ICudaEngine执行时GPU最大需要的空间
        builder.max_batch_size = max_batch_size        # 执行时最大可以使用的batchsize
        builder.fp16_mode = fp16_mode
        config.max_workspace_size = 1 << 30            # 1G

        # 动态输入profile优化
        profile = builder.create_optimization_profile()
        profile.set_shape("input", (1, 3, 224, 224), (8, 3, 224, 224), (8, 3, 224, 224))
        config.add_optimization_profile(profile)

        # 解析onnx文件，填充计算图
        if not os.path.exists(onnx_file_path):
            quit("ONNX file {} not found!".format(onnx_file_path))
        print('loading onnx file from path {} ...'.format(onnx_file_path))
        with open(onnx_file_path, 'rb') as model:
            print("Begining onnx file parsing")
            if not parser.parse(model.read()):         # 解析onnx文件
                print('ERROR: Failed to parse the ONNX file.')
                for error in range(parser.num_errors):
                    print(parser.get_error(error))     # 打印解析错误日志
                return None

        last_layer = network.get_layer(network.num_layers - 1)
        # Check if last layer recognizes it's output
        if not last_layer.get_output(0):
            # If not, then mark the output using TensorRT API
            network.mark_output(last_layer.get_output(0))
        print("Completed parsing of onnx file")

        # 使用builder创建CudaEngine
        print("Building an engine from file{}' this may take a while...".format(onnx_file_path))
        #engine=builder.build_cuda_engine(network)    # 非动态输入使用
        engine=builder.build_engine(network, config)  # 动态输入使用
        print("Completed creating Engine")
        if save_engine:
            with open(engine_file_path, 'wb') as f:
                f.write(engine.serialize())
        return engine


def allocate_buffers(engine):
    inputs, outputs, bindings = [], [], []
    stream = cuda.Stream()
    for binding in engine:
        size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size # 非动态输入
        # size = trt.volume(engine.get_binding_shape(binding))                       # 动态输入
        size = abs(size)                                # 上面得到的size(0)可能为负数，会导致OOM
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        host_mem = cuda.pagelocked_empty(size, dtype)   # 创建锁业内存
        device_mem = cuda.mem_alloc(host_mem.nbytes)    # cuda分配空间
        bindings.append(int(device_mem))                # binding在计算图中的缓冲地址
        if engine.binding_is_input(binding):
            inputs.append(HostDeviceMem(host_mem, device_mem))
        else:
            outputs.append(HostDeviceMem(host_mem, device_mem))

    return inputs, outputs, bindings, stream

def inference(context, bindings, inputs, outputs, stream, batch_size=1):
    # Transfer data from CPU to the GPU.
    [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
    # Run inference.
    # 如果创建network时显式指定了batchsize，使用execute_async_v2, 否则使用execute_async
    context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
    # Transfer predictions back from the GPU.
    [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
    # gpu to cpu
    # Synchronize the stream
    stream.synchronize()
    # Return only the host outputs.
    return [out.host for out in outputs]

def postprocess_the_outputs(h_outputs, shape_of_output):
    h_outputs = h_outputs.reshape(*shape_of_output)
    return h_outputs