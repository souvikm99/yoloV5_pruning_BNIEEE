# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Validate a trained YOLOv5 detection model on a detection dataset

Usage:
    $ python val.py --weights yolov5s.pt --data coco128.yaml --img 640

Usage - formats:
    $ python val.py --weights yolov5s.pt                 # PyTorch
                              yolov5s.torchscript        # TorchScript
                              yolov5s.onnx               # ONNX Runtime or OpenCV DNN with --dnn
                              yolov5s.xml                # OpenVINO
                              yolov5s.engine             # TensorRT
                              yolov5s.mlmodel            # CoreML (macOS-only)
                              yolov5s_saved_model        # TensorFlow SavedModel
                              yolov5s.pb                 # TensorFlow GraphDef
                              yolov5s.tflite             # TensorFlow Lite
                              yolov5s_edgetpu.tflite     # TensorFlow Edge TPU
                              yolov5s_paddle_model       # PaddlePaddle
"""
import matplotlib.pyplot as plt
import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import yaml

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH

ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH

from models.yolo import Detect, Model
from val import run as valrun
from yolo_pruned import ModelPruned



from models.common import DetectMultiBackend, Bottleneck, Concat
from pruned_common import *
from utils.dataloaders import create_dataloader
from utils.general import (LOGGER, check_dataset, check_img_size, check_requirements, check_yaml,
                           colorstr, increment_path, print_args)
from utils.torch_utils import select_device


# In the provided function gather_bn_weights, the Batch Normalization (BN) layer weights are collected 
# by iterating over the module_list, which contains the BN layers.

# First, it calculates the size of each BN layer's weight tensor and stores it in the size_list.
# Then, it initializes a tensor bn_weights of the appropriate size to store all the BN layer weights.
# It iterates over the BN layers, copying the absolute values of their weights into the bn_weights tensor 
# at the appropriate indices.
# Finally, it returns the bn_weights tensor containing all the BN layer weights.

def gather_bn_weights(module_list):
    size_list = [idx.weight.data.shape[0] for idx in module_list.values()]
    bn_weights = torch.zeros(sum(size_list))
    index = 0
    for i, idx in enumerate(module_list.values()):
        size = size_list[i]
        bn_weights[index:(index + size)] = idx.weight.data.abs().clone()
        index += size

    #save the bn weights for readanility 
    bn_weights_file_path = 'checking_data/bn_weights.txt'
    with open(bn_weights_file_path, 'w') as f:
        for weight in bn_weights.numpy():
            f.write(f"{weight}\n")

    return bn_weights


# The function obtain_bn_mask computes a binary mask for the Batch Normalization 
# (BN) layer weights based on a threshold thre. Here's how it works:

# The threshold thre is moved to the specified device.
# The absolute values of the BN layer weights (bn_module.weight.data.abs()) are compared with the threshold.
# Values greater than or equal to the threshold are set to 1 (True) and others to 0 (False), creating a binary mask.
# The mask is returned as a float tensor.
# This function essentially generates a mask indicating which weights of the BN layer are above 
# the threshold and should be retained.


def obtain_bn_mask(bn_module, thre, device):
    # thre = thre.cuda()
    thre = thre.to(device)
    mask = bn_module.weight.data.abs().ge(thre).float()

    bn_mask_file_name = f'checking_data/bn_masks/{bn_module}_bn_mask.txt'
        # Save the mask to a file in a human-readable format
    with open(bn_mask_file_name, 'w') as f:
        # Convert the mask to a numpy array for easy iteration
        mask_array = mask.cpu().numpy()
        for value in mask_array:
            # Write each mask value to the file, converting float to int for readability
            f.write(f"{int(value)}\n")

    return mask


#######CUSTOM FUNCTIONS###########
def save_layerwise_abs_weights_to_file(model, file_path):
    with open(file_path, 'w') as file:
        for name, module in model.named_modules():
            # Check if the layer has the 'weight' attribute (e.g., Convolutional and BatchNorm layers)
            if hasattr(module, 'weight') and module.weight is not None:
                # Get the absolute values of the weights
                abs_weights = module.weight.data.abs()
                
                # Write layer name and its absolute weights to the file
                file.write(f"Layer: {name}\n")
                file.write("Absolute weights:\n")
                abs_weights_str = str(abs_weights.cpu().numpy())
                file.write(abs_weights_str + "\n\n")



def save_bn_layers_absolute_weights(model, filepath):
    with open(filepath, 'w') as f:
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.BatchNorm2d):
                # Obtain the absolute values of the BN weights
                abs_weights = module.weight.data.abs()
                # Convert to a numpy array and flatten for writing
                abs_weights_numpy = abs_weights.cpu().numpy().flatten()
                
                # Write the layer name
                f.write(f"Layer: {name}\n")
                # Write the absolute weights, line by line
                for weight in abs_weights_numpy:
                    f.write(f"{weight:.6f}\n")
                f.write("\n")  # Add a newline for readability between layers


def save_bn_associated_layer_params(model, filepath):
    with open(filepath, 'w') as f:
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.BatchNorm2d):
                # Find the layer name immediately preceding the BN layer, assuming a naming convention
                # This might need adjustment based on the specific architecture
                associated_layer_name = name.rsplit('.', 1)[0] + '.conv'  # Adjust based on your model's naming convention
                associated_layer = dict(model.named_modules()).get(associated_layer_name, None)
                
                # Proceed if the associated layer is found
                if associated_layer:
                    # Calculate total parameters for the associated layer
                    total_params = sum(p.numel() for p in associated_layer.parameters())
                    
                    # Write information to the file
                    f.write(f"BN Layer: {name}\n")
                    f.write(f"Associated Layer: {associated_layer_name}, Total Parameters: {total_params}\n\n")


def save_total_model_parameters(model, filepath):
    total_params = sum(p.numel() for p in model.parameters())
    with open(filepath, 'w') as f:
        f.write(f"Total Parameters in Model: {total_params}\n")
    return total_params

def calculate_percentage_of_parameters(total_params, percent):
    percentage_of_params = (percent / 100.0) * total_params
    return percentage_of_params

def accumulate_parameters_to_percentage(model, percent_target, filepath):
    # Calculate total parameters and target number of parameters
    total_params = sum(p.numel() for p in model.parameters())
    target_params = total_params * (percent_target / 100.0)

    # Function to accumulate parameters and layers
    def accumulate_layers_params(layers):
        accum_params = 0
        selected_layers = []
        for name, params in layers:
            layer_params = params.numel()
            if accum_params + layer_params > target_params:
                break  # Stop if next layer exceeds target
            accum_params += layer_params
            selected_layers.append((name, layer_params))
        return selected_layers, accum_params

    # Collect layers and their parameters
    layers_with_params = [(name, param) for name, param in model.named_parameters()]

    # Accumulate parameters and layers
    selected_layers, accum_params = accumulate_layers_params(layers_with_params)

    # Save to file
    with open(filepath, 'w') as file:
        file.write(f"Target Percentage: {percent_target}% of Total Parameters: {target_params} (Approx.)\n")
        file.write(f"Selected layers and parameters to reach ~{percent_target}%:\n")
        for name, params in selected_layers:
            file.write(f"{name}: {params} params\n")
        file.write(f"\nTotal Selected Parameters: {accum_params}\n")
        file.write(f"Total Model Parameters: {total_params}\n")

    print(f"Details saved to {filepath}")

def save_pruned_channels_info(maskbndict, filepath):
    with open(filepath, 'w') as file:
        total_pruned_channels = 0
        for bn_name, mask in maskbndict.items():
            pruned_channels = [i for i, value in enumerate(mask.tolist()) if value == 0]
            num_pruned_channels = len(pruned_channels)
            if num_pruned_channels > 0:
                file.write(f"{bn_name}: Pruned {num_pruned_channels} channels\n")
                file.write(f"Pruned Channels Indexes: {pruned_channels}\n\n")
                total_pruned_channels += num_pruned_channels
        file.write(f"Total Pruned Channels across all BN layers: {total_pruned_channels}\n")

#save connected convolution layers channels
def save_bn_conv_params(model, filepath):
    total_params = 0  # Initialize total parameters counter
    
    with open(filepath, 'w') as file:
        for name, module in model.named_modules():
            if isinstance(module, nn.BatchNorm2d):
                # Assuming the naming convention Conv -> BN
                conv_name = name.rsplit('.', 1)[0] + '.conv'  # This might need adjustment
                conv_layer = dict(model.named_modules()).get(conv_name, None)
                
                if conv_layer and isinstance(conv_layer, nn.Conv2d):
                    # Calculate the number of parameters in the Conv layer
                    # Parameters = out_channels * (in_channels * kernel_height * kernel_width)
                    conv_params = conv_layer.out_channels * (conv_layer.in_channels * 
                                                             conv_layer.kernel_size[0] * conv_layer.kernel_size[1])
                    total_params += conv_params  # Accumulate the total parameters
                    
                    # Write detailed information about the layer
                    file.write(f"BN Layer: {name}, Connected Conv Layer: {conv_name}, Number of Parameters: {conv_params}\n")
        
        # After iterating through all layers, write the total parameters
        file.write(f"\nTotal Number of Parameters (from connected Conv layers to BN layers): {total_params}\n")


#######END CUSTOM FUNCTIONS###########






@torch.no_grad()
def run_prune(data,
              weights=None,  # model.pt path(s)
              cfg='models/yolov5s.yaml',
              percent=0,
              batch_size=32,  # batch size
              imgsz=640,  # inference size (pixels)
              task='val',  # train, val, test, speed or study
              device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
              workers=8,  # max dataloader workers (per RANK in DDP mode)
              single_cls=False,  # treat as single-class dataset
              save_txt=False,  # save results to *.txt
              project=ROOT / 'runs/val',  # save to project/name
              name='exp',  # save to project/name
              exist_ok=False,  # existing project/name ok, do not increment
              dnn=False,  # use OpenCV DNN for ONNX inference
              model=None,
              ):
    # Initialize/load model and set device
    device = select_device(device, batch_size=batch_size)

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Load model, the fuse here must be turned off, otherwise the BN layer will be fused into the Conv layer, 
    # and the information of the BN layer can no longer be counted.
    
    model = DetectMultiBackend(weights, device=device, dnn=dnn, fuse=False)
    stride, pt, jit, engine = model.stride, model.pt, model.jit, model.engine
    imgsz = check_img_size(imgsz, s=stride)  # check image size
    data = check_dataset(data)  # check

    # Configure
    model = model.model
    model.eval()

    # After loading and configuring your model
    save_layerwise_abs_weights_to_file(model, 'checking_data/layerwise_abs_weights.txt')
    save_bn_layers_absolute_weights(model, 'checking_data/sorted_bn_weights.txt')
    # Use this function after model configuration
    save_bn_associated_layer_params(model, 'checking_data/bn_associated_layer_params.txt')
    # Use this function after model configuration
    # Calculate and save total parameters to a file
    total_params = save_total_model_parameters(model, 'checking_data/total_model_parameters.txt')
    # Calculate what the passed percentage of total parameters would be
    percent_params = calculate_percentage_of_parameters(total_params, opt.percent_new)
    print(f"{opt.percent_new}% of Total Parameters: {percent_params:.0f}")
    # Use this function to calculate and save layers and parameters
    accumulate_parameters_to_percentage(model, opt.percent_new, 'checking_data/selected_layers_params.txt')
    #layer wise channel parameter for each connected bn layer
    save_bn_conv_params(model, 'checking_data/bn_conv_params.txt')

    # =========================================== prune model ====================================#
    # print("model.module_list:",model.named_children())
    #maybe this should be module_list
    model_list = {}
    ignore_bn_list = []

    # Count the modules that do not need pruning
    for i, layer in model.named_modules():
        if isinstance(layer, Bottleneck):
            if layer.add:
                ignore_bn_list.append(i.rsplit(".", 2)[0] + ".cv1.bn")
                ignore_bn_list.append(i + '.cv1.bn')
                ignore_bn_list.append(i + '.cv2.bn')
                print(f'{i.rsplit(".", 2)[0] + ".cv1.bn"} | {i + ".cv1.bn"} | {i + ".cv2.bn"}')
    # print(f"ignore bn list: {ignore_bn_list}")
    for i, layer in model.named_modules():
        if isinstance(layer, torch.nn.BatchNorm2d):
            if i not in ignore_bn_list:
                model_list[i] = layer

    ################################################
    ################################################
    # Save model list to a text file
    with open('checking_data/model_list.txt', 'w') as f:
        for key, value in model_list.items():
            f.write(f"{key}: {value}\n")

    # Save ignore_bn_list to a text file
    with open('checking_data/ignore_bn_list.txt', 'w') as f:
        for item in ignore_bn_list:
            f.write(f"{item}\n")
    ################################################
    ################################################


    # Collecting BN layer names and the number of weights in each layer
    bn_layer_names = list(model_list.keys())
    bn_layer_values = list(model_list.values())
    bn_weights_counts = [len(model_list[bn_name].weight.data) for bn_name in bn_layer_names]
    print(f'bn_weights_counts : {bn_weights_counts}')
    total_model_list_weight_count = len(bn_weights_counts)

    # with open('bn_layers_name_val.txt', 'w') as f:
    #  for key, value in model_list.items():
    #          # Write the layer name
    #          f.write(f"Layer: {key}\n")
    #          # Write the absolute weights, line by line
    #          f.write(f"{value}\n")
    #          f.write("\n")  # Add a newline for readability between layers

    # Plotting
    plt.figure(figsize=(12, 8))
    plt.bar(bn_layer_names, bn_weights_counts, color='skyblue')
    plt.title(f'Number of Weights in Each BN Layer of selected layers')
    plt.xlabel('BN Layer')
    plt.ylabel('Number of Weights')
    plt.xticks(rotation=90)  # Rotate the x-axis labels for better readability
    plt.tight_layout()  # Adjust layout to make room for the rotated x-axis labels

    # Save the figure
    plt.savefig('checking_data/bn_layers_weights_count.png')
    plt.show()


    # Collect the weights of all BN layers that need pruning
    bn_weights = gather_bn_weights(model_list)
    sorted_bn = torch.sort(bn_weights)[0]

    # Avoid the highest threshold that would prune all channels 
    # (the smallest maximum value of gamma for each BN layer is the threshold upper limit).

    #To ensure that at least one channel remains in each BN layer, first check the maximum value of the BN layers, 
    # then take the smallest value from these maximums. 
    # This ensures that each BN layer has at least one channel that is not removed.
    highest_thre = []
    for bnlayer in model_list.values():
        highest_thre.append(bnlayer.weight.data.abs().max().item())
    # print("highest_thre:",highest_thre)
    highest_thre = min(highest_thre)
    # Find the percentage corresponding to the index of highest_thre
    percent_limit = (sorted_bn == highest_thre).nonzero()[0, 0].item() / len(bn_weights)

    print(f'Suggested Gamma threshold should be less than {highest_thre:.4f}.')
    print(f'The corresponding prune ratio is {percent_limit:.3f}')

    assert percent_limit > percent, f'The pruning ratio should not exceed {percent_limit * 100:.3f}%!'

    # model_copy = deepcopy(model)
    # Obtain the pruning threshold according to the specified percentage
    #######
    print(f"SORTED BN LENGTH: {len(sorted_bn)}")
    #######
    thre_index = int(len(sorted_bn) * percent)
    thre = sorted_bn[thre_index]

    #######
    print(f"Threshold index: len(sorted_bn) * percent = thre_index ")
    print(f"Threshold index: {len(sorted_bn)} * {percent} = {thre_index} ")
    print(f"thre: {thre}")
    #######
    print(f'Gamma value that less than {thre:.4f} are set to zero!')
    print("=" * 94)
    print(f"|\t{'layer name':<25}{'|':<10}{'origin channels':<20}{'|':<10}{'remaining channels':<20}|")


    # Plot the BN layer weights and save the figure
    plt.figure(figsize=(10, 5))
    plt.plot(sorted_bn.cpu().numpy(), label='BN weights')
    plt.axvline(x=thre_index, color='r', linestyle='--', label=f'Threshold at {percent*100}%')
    plt.title('BN Layer Weights Distribution After Sorting')
    plt.xlabel('Index')
    plt.ylabel('Weight Value')
    plt.legend()

    # Save the figure
    plt.savefig('checking_data/bn_weights_distribution_new.png')

    # ============================== save pruned model config yaml =================================#
    remain_num = 0
    modelstate = model.state_dict()
    pruned_yaml = {}
    nc = model.model[-1].nc
    with open(cfg, encoding='ascii', errors='ignore') as f:
        model_yamls = yaml.safe_load(f)  # model dict
    pruned_yaml["nc"] = model.model[-1].nc
    pruned_yaml["depth_multiple"] = model_yamls["depth_multiple"]
    pruned_yaml["width_multiple"] = model_yamls["width_multiple"]
    pruned_yaml["anchors"] = model_yamls["anchors"]
    anchors = model_yamls["anchors"]
    pruned_yaml["backbone"] = [
        [-1, 1, Conv, [64, 6, 2, 2]],  # 0-P1/2
        [-1, 1, Conv, [128, 3, 2]],  # 1-P2/4
        [-1, 3, C3, [128]],
        [-1, 1, Conv, [256, 3, 2]],  # 3-P3/8
        [-1, 6, C3, [256]],
        [-1, 1, Conv, [512, 3, 2]],  # 5-P4/16
        [-1, 9, C3, [512]],
        [-1, 1, Conv, [1024, 3, 2]],  # 7-P5/32
        [-1, 3, C3, [1024]],
        [-1, 1, SPPF, [1024, 5]],  # 9
    ]
    pruned_yaml["head"] = [
        [-1, 1, Conv, [512, 1, 1]],
        [-1, 1, nn.Upsample, [None, 2, 'nearest']],
        [[-1, 6], 1, Concat, [1]],  # cat backbone P4
        [-1, 3, C3, [512, False]],  # 13

        [-1, 1, Conv, [256, 1, 1]],
        [-1, 1, nn.Upsample, [None, 2, 'nearest']],
        [[-1, 4], 1, Concat, [1]],  # cat backbone P3
        [-1, 3, C3, [256, False]],  # 17 (P3/8-small)

        [-1, 1, Conv, [256, 3, 2]],
        [[-1, 14], 1, Concat, [1]],  # cat head P4
        [-1, 3, C3, [512, False]],  # 20 (P4/16-medium)

        [-1, 1, Conv, [512, 3, 2]],
        [[-1, 10], 1, Concat, [1]],  # cat head P5
        [-1, 3, C3, [1024, False]],  # 23 (P5/32-large)

        [[17, 20, 23], 1, Detect, [nc, anchors]],  # Detect(P3, P4, P5)
    ]

    # ============================================================================== #
    maskbndict = {}

    # Output the pruning information for each layer
    for bn_name, bn_layer in model.named_modules():
        if isinstance(bn_layer, nn.BatchNorm2d):
            if bn_name in ignore_bn_list:
                # mask = torch.ones(bn_layer.weight.data.size()).cuda()
                mask = torch.ones(bn_layer.weight.data.size()).to(device)
            else:
                mask = obtain_bn_mask(bn_layer, thre, device)

            maskbndict[bn_name] = mask
            # Number of remaining channels in the current layer
            layer_remain = int(mask.sum())
            assert layer_remain > 0, "Current remaining channel must greater than 0!!! " \
                                     "please set prune percent to lower thesh, or you can retrain a more sparse model..."
            # Calculate the total number of remaining channels
            remain_num += layer_remain

            # Set the weights and biases of the BN layers that need pruning to zero
            bn_layer.weight.data.mul_(mask)
            bn_layer.bias.data.mul_(mask)
            print(f"|\t{bn_name:<25}{'|':<10}{bn_layer.weight.data.size()[0]:<20}{'|':<10}{layer_remain:<20}|")

    print("=" * 94)

    # To avoid overlapping prints between the previous print statements and the subsequent ModelPruned prints, 
    # sleep for 1 second here to allow the previous content to finish.
    time.sleep(1)

    # Call this function after you have applied all the masks to the BN layers
    save_pruned_channels_info(maskbndict, 'checking_data/saved_pruned_channels_info.txt')

    # Reconstruct the YOLO model based on the mask information.
    # pruned_model = ModelPruned(maskbndict=maskbndict, cfg=pruned_yaml, ch=3).cuda()
    pruned_model = ModelPruned(maskbndict=maskbndict, cfg=pruned_yaml, ch=3).to(device)
    for m in pruned_model.modules():
        if type(m) in [nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU, Detect, Model]:
            m.inplace = True  # pytorch 1.7.0 compatibility
        elif type(m) is Conv:
            m._non_persistent_buffers_set = set()  # pytorch 1.6.0 compatibility

    # The `from_to_map` stores information about which layer's input comes from which layer, 
    # for example: `{'model.2.cv3.bn': ['model.2.m.0.cv2.bn', 'model.2.cv2.bn']}`.    
    from_to_map = pruned_model.from_to_map
    pruned_model_state = pruned_model.state_dict()
    assert pruned_model_state.keys() == modelstate.keys()
    # ===================================# Process input and output channels==================================================== #
    changed_state = []
    for ((layername, layer), (pruned_layername, pruned_layer)) in zip(model.named_modules(),
                                                                      pruned_model.named_modules()):
        assert layername == pruned_layername
        # model.24 is the Detect layer, which contains three Conv layers
        if isinstance(layer, nn.Conv2d) and not layername.startswith("model.24"):
            convname = layername[:-4] + "bn"
            if convname in from_to_map.keys():
                former = from_to_map[convname]
                if isinstance(former, str):
                    out_idx = np.squeeze(np.argwhere(np.asarray(maskbndict[layername[:-4] + "bn"].cpu().numpy())))
                    in_idx = np.squeeze(np.argwhere(np.asarray(maskbndict[former].cpu().numpy())))
                    w = layer.weight.data[:, in_idx, :, :].clone()  # Process input channels

                    if len(w.shape) == 3:  # remain only 1 channel.
                        w = w.unsqueeze(1)
                    w = w[out_idx, :, :, :].clone()  # Process output channels

                    pruned_layer.weight.data = w.clone()  # Assign values again
                    changed_state.append(layername + ".weight")
                if isinstance(former, list):
                    orignin = [modelstate[i + ".weight"].shape[0] for i in former]
                    formerin = []
                    for it in range(len(former)):
                        name = former[it]
                        tmp = [i for i in range(maskbndict[name].shape[0]) if maskbndict[name][i] == 1]
                        if it > 0:
                            tmp = [k + sum(orignin[:it]) for k in tmp]
                        formerin.extend(tmp)
                    out_idx = np.squeeze(np.argwhere(np.asarray(maskbndict[layername[:-4] + "bn"].cpu().numpy())))
                    w = layer.weight.data[out_idx, :, :, :].clone()
                    pruned_layer.weight.data = w[:, formerin, :, :].clone()
                    changed_state.append(layername + ".weight")
            else:

            # Process model.0.Conv, as this layer does not have input from the previous 
            # layer and will not appear in from_to_map

                out_idx = np.squeeze(np.argwhere(np.asarray(maskbndict[layername[:-4] + "bn"].cpu().numpy())))
                w = layer.weight.data[out_idx, :, :, :].clone()
                assert len(w.shape) == 4
                pruned_layer.weight.data = w.clone()
                changed_state.append(layername + ".weight")

        if isinstance(layer, nn.BatchNorm2d):
            out_idx = np.squeeze(np.argwhere(np.asarray(maskbndict[layername].cpu().numpy())))
            pruned_layer.weight.data = layer.weight.data[out_idx].clone()
            pruned_layer.bias.data = layer.bias.data[out_idx].clone()
            pruned_layer.running_mean = layer.running_mean[out_idx].clone()
            pruned_layer.running_var = layer.running_var[out_idx].clone()
            changed_state.append(layername + ".weight")
            changed_state.append(layername + ".bias")
            changed_state.append(layername + ".running_mean")
            changed_state.append(layername + ".running_var")
            changed_state.append(layername + ".num_batches_tracked")

        # Handle the last Detect layer separately
        if isinstance(layer, nn.Conv2d) and layername.startswith("model.24"):
            former = from_to_map[layername]
            in_idx = np.squeeze(np.argwhere(np.asarray(maskbndict[former].cpu().numpy())))
            pruned_layer.weight.data = layer.weight.data[:, in_idx, :, :]
            pruned_layer.bias.data = layer.bias.data
            changed_state.append(layername + ".weight")
            changed_state.append(layername + ".bias")

    pruned_model.eval()
    pruned_model.names = model.names
    # =============================================================================================== #
    torch.save({"model": model}, os.path.join(save_dir, "orign_model.pt"))
    torch.save({"model": pruned_model}, os.path.join(save_dir, "pruned_model.pt"))
    LOGGER.info(f'Pruned model weights saved at {os.path.join(save_dir, "pruned_model.pt")}')

    # Start validating the metrics of the pruned model
    model = pruned_model
    #after prunning layer wise channel parameter for each connected bn layer
    save_bn_conv_params(model, 'checking_data/after_bn_conv_params.txt')
    model.to(device).eval()
    # model.cuda().eval()


    # Create dataloader
    pad = 0.0 if task in ('speed', 'benchmark') else 0.5
    rect = False if task == 'benchmark' else pt  # square inference for benchmarks
    task = task if task in ('train', 'val', 'test') else 'val'  # path to train/val/test images
    dataloader = create_dataloader(data[task],
                                   imgsz,
                                   batch_size,
                                   stride,
                                   single_cls,
                                   pad=pad,
                                   rect=rect,
                                   workers=workers,
                                   prefix=colorstr(f'{task}: '))[0]

    data_dict = check_dataset(data)  # check if None
    LOGGER.info('After Pruning.....')
    valrun(
        data=data_dict,
        model=model,
        dataloader=dataloader,
        batch_size=opt.batch_size,
        imgsz=opt.imgsz,
        workers=opt.workers,
        half=opt.half,
        plots=False
    )


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default=ROOT / 'data/mask.yaml', help='dataset.yaml path')
    parser.add_argument('--cfg', type=str, default=ROOT / 'models/yolov5n_mask.yaml', help='model.yaml path')
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'runs/train/exp6/weights/last.pt',
                        help='model path(s)')
    parser.add_argument('--percent', type=float, default=0.4, help='prune percentage')
    parser.add_argument('--batch-size', type=int, default=64, help='batch size')
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=320, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.001, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.6, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=300, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--workers', type=int, default=0, help='max dataloader workers (per RANK in DDP mode)')
    parser.add_argument('--project', default=ROOT / 'runs/val', help='save to project/name')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    parser.add_argument("--percent_new", type=float, default=0, help="Percentage of total parameters")

    opt = parser.parse_args()
    opt.data = check_yaml(opt.data)  # check YAML
    print_args(vars(opt))
    return opt


def main(opt):
    check_requirements(exclude=('tensorboard', 'thop'))
    if opt.conf_thres > 0.001:  # https://github.com/ultralytics/yolov5/issues/1466
        LOGGER.info(f'WARNING ⚠️ confidence threshold {opt.conf_thres} > 0.001 produces invalid results')

    LOGGER.info('Before Pruning.....')

    # valrun(
    #     data=opt.data,
    #     weights=opt.weights,
    #     batch_size=opt.batch_size,
    #     imgsz=opt.imgsz,
    #     workers=opt.workers,
    #     project=opt.project,
    #     half=opt.half
    # )

    LOGGER.info('Pruning.....')
    run_prune(
        data=opt.data,
        weights=opt.weights,
        cfg=opt.cfg,
        percent=opt.percent,
        batch_size=opt.batch_size,
        imgsz=opt.imgsz,
        workers=opt.workers,
        project=opt.project
    )


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
