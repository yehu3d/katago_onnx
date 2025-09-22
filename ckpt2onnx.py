#!/usr/bin/python3
import argparse
import load_model
import torch
import torch.nn
import torch.optim
import torch.distributed
import torch.multiprocessing
from load_model import load_model, load_model_state_dict
#python ckpt2onnx.py --i_ckpt 28bnbt.ckpt --o_onnx 28bnbt.onnx

class PrunedModel(torch.nn.Module):
    def __init__(self, original_model):
        super(PrunedModel, self).__init__()
        self.original_model = original_model

    def forward(self, input_spatial, input_global):
        outputs = self.original_model(input_spatial, input_global)
        pruned_outputs = tuple([outputs[0][i] for i in [0, 1, 2, 3, 4]])
        return pruned_outputs

def main():
    
    parser = argparse.ArgumentParser(description='Convert CKPT model to ONNX format')
    parser.add_argument('--i_ckpt', type=str, required=True, help='Input CKPT model file path')
    parser.add_argument('--o_onnx', type=str, required=True, help='Output ONNX file path')
    
    args = parser.parse_args()
    

    model, swa_model, _ = load_model(args.input, 0, device='cuda', pos_len=19, verbose=True)
    model = PrunedModel(model)

    dynamic_axes = {
        "input_spatial": {0: "batch_size"},
        "input_global": {0: "batch_size"},
        "out_policy": {0: "batch_size"},
        "out_value": {0: "batch_size"},
        "out_miscvalue": {0: "batch_size"},
        "out_moremiscvalue": {0: "batch_size"},
        "out_ownership": {0: "batch_size"},
    }
    # 0 is dynamic batchs
    
    c_global_input_data = torch.randn(1, 19, device='cuda')
    c_bin_input_data = torch.randn(1, 22, 19, 19, device='cuda')
    
    torch.onnx.export(model, 
                      (c_bin_input_data, c_global_input_data), 
                      args.output,
                      input_names=["input_spatial", "input_global"],
                      output_names=[
                          "out_policy",
                          "out_value",
                          "out_miscvalue",
                          "out_moremiscvalue",
                          "out_ownership"],
                      verbose=False,
                      opset_version=19,
                      dynamic_axes=dynamic_axes)
    
    print(f"Model successfully converted from {args.input} to {args.output}")

if __name__ == "__main__":
    main()
