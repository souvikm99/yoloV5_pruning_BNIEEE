import torch
from models.yolo import Model  # Adjust this import based on your directory structure

# Path to your YOLOv5 model .pt file
model_path = 'Pruning_exp/YolovS_mycropsdata_3202/weights/best.pt'

# Load the model
model = torch.load(model_path)['model'].float()  # Ensure model is in float format
model.eval()  # Set model to evaluation mode

# Prepare to count parameters
total_params = 0
layer_params_list = []

# Iterate through model layers
for name, parameter in model.named_parameters():
    # Count the parameters for this layer
    param = parameter.numel()
    # Add to the total count
    total_params += param
    # Record the layer's parameter count
    layer_params_list.append(f"{name}: {param}")

# Add total parameters to the list
layer_params_list.append(f"Total number of parameters in the model: {total_params}")

# Output layer-wise parameter count and total
for line in layer_params_list:
    print(line)

# Saving to a text file
with open('checking_data/model_parameters_count.txt', 'w') as file:
    for line in layer_params_list:
        file.write(line + '\n')

print("Layer-wise parameter count saved to model_parameters_count.txt")
