import torch.quantization.observer as observer
import torch
from torch.quantization.quantize_fx import prepare_qat_fx,convert_fx
import copy


#将模型转换为QAT版本模型，其他训练过程与正常训练类似
def qat_version_model(model):
    qconfig_dict = {
        # Global Config
        "":torch.ao.quantization.get_default_qat_qconfig('qnnpack'), #全局量化配置

        # # Disable by layer-name
        # "module_name": [(m, None) for m in disable_layers],

        # Or disable by layer-type
        # "object_type": [
        #     (PositionalEmbedding, None),   #想要跳过量化层，可以设置为None
        #     (torch.nn.Softmax, softmax_qconfig), #指定与全局量化配置不同的量化配置
        #     ......
        # ],
    }
    model_to_quantize = copy.deepcopy(model)
    example_inputs = (torch.randn(1, 3, 224, 224),)
    model_fp32_prepared = prepare_qat_fx(
        model_to_quantize, qconfig_dict, example_inputs=example_inputs)
    return model_fp32_prepared