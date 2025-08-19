import torch
import argparse
import os
from collections import OrderedDict
# Make sure the arch file is in the same directory or accessible in PYTHONPATH
from gen3_vsr_arch import GEN3VSR, GEN3VSR_ET

def get_info(state: OrderedDict) -> dict:
    """
    Obtains model hyperparameters from the state dictionary.
    """
    mid_channels = state['fea_conv.weight'].shape[0]
    
    # Count the number of conv layers in the middle block
    num_middle_convs = sum(1 for key in state if key.startswith('middle_convs') and key.endswith('.weight'))
    num_blocks = num_middle_convs

    # Infer scale from the output channels of tconv3
    out_ch_tconv3 = state['tconv3.weight'].shape[0]
    # Assuming out_channels is 3
    scale = int((out_ch_tconv3 / 3) ** 0.5)

    return {
        'mid_channels': mid_channels,
        'num_blocks': num_blocks,
        'scale': scale,
    }

def model_allclose(model1: GEN3VSR, model2: GEN3VSR_ET) -> bool:
    """
    Checks if the outputs of the two models are numerically very close.
    """
    # Generate a random input tensor
    input_tensor = torch.rand(1, 3, 64, 64) * 255

    # Set models to evaluation mode
    model1.eval()
    model2.eval()

    # Get outputs from both models
    # Note: model1 includes a clamp, model2 does not. We compare against the clamped output.
    out1 = model1(input_tensor)
    out2 = torch.clamp(model2(input_tensor), min=0., max=255.)

    # Calculate the maximum absolute difference
    max_diff = torch.max((out1 - out2).abs()).item()
    print(f"Maximum absolute difference between models: {max_diff}")
    
    # Allow for a small floating point tolerance
    return max_diff < 2e-4

def convert(state: OrderedDict) -> OrderedDict:
    """
    Converts the GEN3VSR state_dict to a plain network state_dict (GEN3VSR_ET).
    This removes the feat_skip connection by modifying the subsequent convolution (tconv1).
    """
    # Handle nested state dicts (e.g., from BasicSR)
    if 'params' in state:
        model_state = state['params']
    else:
        model_state = state

    # Get model hyperparameters from the state_dict
    info = get_info(model_state)
    print(f"Detected model parameters: {info}")

    # Instantiate the original and ET models
    model1 = GEN3VSR(**info)
    model1.load_state_dict(model_state)
    model2 = GEN3VSR_ET(**info)

    # Copy the state dict to modify it
    converted_state = model_state.copy()

    # --- Reparameterization Step ---
    # The operation is middle_convs(x) + x, followed by tconv1.
    # We replace the 'add' with a 'concat' and fuse the logic into tconv1.
    # The new tconv1 will have weights [W, W] to simulate W(A+B) = WA + WB.
    
    print("Target layer for transformation: 'tconv1'")
    
    original_tconv1_weight = model_state['tconv1.weight']
    
    # Concatenate the weights along the input channel dimension (dim=1)
    reparameterized_weight = torch.cat([original_tconv1_weight, original_tconv1_weight], dim=1)
    
    # Update the weight in the new state dictionary
    converted_state['tconv1.weight'] = reparameterized_weight
    
    # The bias of tconv1 does not need to be changed.
    # W_new @ cat(A,B) + b = (W@A + W@B) + b
    # W_orig @ (A+B) + b = (W@A + W@B) + b
    # The biases are equivalent.

    # Load the converted state into the ET model
    model2.load_state_dict(converted_state)

    # --- Verification ---
    # Ensure the transformation was successful and outputs are identical
    assert model_allclose(model1, model2), "Model outputs do not match after conversion!"
    print("Verification successful: Model outputs match.")

    return converted_state


def main():
    """
    Main function to run the converter from the command line.
    """
    parser = argparse.ArgumentParser(description='GEN3VSR to GEN3VSR_ET Converter')
    parser.add_argument('--input', type=str, required=True, help='Path to the input GEN3VSR model state_dict (.pth file).')
    parser.add_argument('--output', type=str, required=True, help='Path to save the output GEN3VSR_ET model state_dict (.pth file).')
    args = parser.parse_args()

    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Load the original model's state dictionary
    print(f"Loading model from: {args.input}")
    # Ensure weights are loaded to the CPU
    state = torch.load(args.input, map_location=torch.device('cpu'))

    # Perform the conversion
    print("Starting conversion...")
    converted_state = convert(state)

    # Save the new state dictionary
    torch.save(converted_state, args.output)
    print(f"Successfully converted and saved model to: {args.output}")


if __name__ == '__main__':
    main()
