# 模型压缩包

model_compression专注于模型优化，目前支持模型蒸馏、剪枝、量化、部署。用户只需使用默认配置或根据具体需要更改配置文件即可实现模型的优化

![未命名绘图.drawio (1)](.\框架图.png)

## 安装

要求

- torch==1.12.1+cu116
- onnx==1.13.1
- onnxruntime==1.14.1
- tensorrt==8.6.0
- transformers==4.30.2
- datasets==2.12.0

请克隆该存储库并使用pip进行安装

```bash
git clone https://github.com/wytszhuzhu/model_compression.git
pip install -r requirements.txt
cd model_compression
pip install [-e] .
```

-e选项代表安装可编辑版本的库

## 快速浏览

## 量化感知训练

量化感知训练"通常是指在深度学习中使用量化感知器（Quantized Perception）进行训练的一种技术。量化感知器是一种神经网络模型，其核心思想是使用离散的权重和激活值来减少计算和存储需求，从而实现更高效的推理和训练过程。

```python
from model_compression import qat_version_model
import copy
import torch
from torch.quantization.quantize_fx import prepare_qat_fx,convert_fx
import torch.quantization.observer as observer

model = get_model()  #初始化自己模型
qat_model = qat_version_model(model)

# training loop
for i, batch in enumerate(data):
    # do forward procedures
    ...


quantized_model = convert_fx(qat_model)
torch.save(quantized_model.state_dict(), os.path.join(save_dir, "quant.pth"))    
```



## 权重裁剪

权重剪裁（Weight Pruning）是一种用于深度学习模型优化的技术，其目的是通过移除神经网络中不必要或冗余的权重，从而减少模型的计算和存储需求，提高模型的效率和推理速度。

```python
from model_compression_research import IterativePruningConfig, IterativePruningScheduler

model = get_model()  #初始化自己模型
#初始化剪裁配置
pruning_config = IterativePruningConfig(
    pruning_fn="unstructured_magnitude",
    pruning_fn_default_kwargs={"target_sparsity": 0.9}
)
pruning_scheduler = IterativePruningScheduler(model, pruning_config)

# training loop
for i, batch in enumerate(data):
    # do forward procedures
    ...

# 训练结束时删除剪枝部分
pruning_scheduler.remove_pruning() 
```

## 知识蒸馏

知识蒸馏（Knowledge Distillation）是一种用于深度神经网络优化的技术，其目的是将一个复杂的模型的知识（知识来源）转移给一个更简单的模型（知识接收者），从而使得接收者模型能够在相对较低的计算资源下实现与复杂模型接近的性能。

```python
from model_compression_research import TeacherWrapper, DistillationModelWrapper

teacher = get_teacher_trained_model() # 教师模型
student = get_student_model()         # 学生模型

# 使用装饰器包裹学生模型并设置温度系数
teacher = TeacherWrapper(teacher, ce_alpha=0.5, ce_temperature=2.0)
# 初始化学生模型和教师模型
distillation_model = DistillationModelWrapper(student, teacher, alpha_student=0.5)


# training loop
for i, batch in enumerate(data):
    # do forward procedures
    ...
```

## 模型部署

模型部署是将已经训练好的深度学习模型应用到实际生产环境中的过程。在模型部署阶段，我们需要将训练好的模型转换为可以在目标平台上运行的格式，并通过合适的方式进行部署和集成，以实现模型的实时预测和推理。

```python
import time
import onnx
import torch
import torchvision
import onnxruntime
from model_compression import HostDeviceMem, build_engine, allocate_buffers, inference

if __name__ == '__main__':
    onnx_file_path = "resnet50.onnx"
    fp16_mode = False
    max_batch_size = 1
    trt_engine_path = "resnet50.trt"
    
    # 1.创建cudaEngine
    engine = build_engine(onnx_file_path, trt_engine_path, max_batch_size, fp16_mode)
    
    # 2.将引擎应用到不同的GPU上配置执行环境
    context = engine.create_execution_context()
    inputs, outputs, bindings, stream = allocate_buffers(engine)

    # 3.推理
    output_shape = (max_batch_size, 1000)
    dummy_input = np.ones([1, 3, 224, 224], dtype=np.float32)
    inputs[0].host = dummy_input.reshape(-1)

    # 如果是动态输入，需以下设置
    context.set_binding_shape(0, dummy_input.shape)
    
    t1 = time.time()
    trt_outputs = inference(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)
    t2 = time.time()
    # 由于tensorrt输出为一维向量，需要reshape到指定尺寸
    feat = postprocess_the_outputs(trt_outputs[0], output_shape)

    # 4.速度对比
    model = torchvision.models.resnet50(pretrained=True).cuda()
    model = model.eval()
    dummy_input = torch.zeros((1, 3, 224, 224), dtype=torch.float32).cuda()
    t3 = time.time()
    feat_2= model(dummy_input)
    t4 = time.time()
    feat_2 = feat_2.cpu().data.numpy()

    mse = np.mean((feat - feat_2)**2)
    print("TensorRT engine time cost: {}".format(t2-t1))
    print("PyTorch model time cost: {}".format(t4-t3))
    print('MSE Error = {}'.format(mse))
```

输出

```
TensorRT engine time cost: 0.0036957263946533203
PyTorch model time cost: 0.016288280487060547
MSE Error = 0.063191719353199
```

