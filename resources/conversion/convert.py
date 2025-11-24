import torch
import torch.nn as nn
import ai_edge_torch

class e2eNetwork(nn.Module):
    def __init__(self):
        # TODO make sure the layers and activation function match the model you have trained
        super(e2eNetwork, self).__init__()
        self.fc1 = nn.Linear(15, 256)  # Input layer (15 inputs, position error, velocity, attitude (6D), angular velocity)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 4) # Output layer (number of motors)

    def forward(self, x):
        x = torch.relu(self.fc1(x)) # Make sure to add correct activation functions
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x

def convert_network():
    # Load the state dictionary. TODO: Change the model name to match the one you have trained
    state_dict = torch.load("gen_ppo.pth", map_location=torch.device('cpu'))
    # Extract the model state dictionary
    model_state_dict = state_dict["model"]

    # Map the keys to match the e2eNetwork structure
    mapped_state_dict = {
        "fc1.weight": model_state_dict["a2c_network.actor_mlp.0.weight"],
        "fc1.bias": model_state_dict["a2c_network.actor_mlp.0.bias"],
        "fc2.weight": model_state_dict["a2c_network.actor_mlp.2.weight"],
        "fc2.bias": model_state_dict["a2c_network.actor_mlp.2.bias"],
        'fc3.weight': 'a2c_network.actor_mlp.4.weight',  
        'fc3.bias':   'a2c_network.actor_mlp.4.bias',  
        'fc4.weight': 'a2c_network.mu.weight',  
        'fc4.bias':   'a2c_network.mu.bias',      }

    # Initialize the e2eNetwork model
    e2e_model = e2eNetwork()
    try:
        res = e2e_model.load_state_dict(mapped_state_dict, strict=False)
        print('missing_keys:', res.missing_keys)
        print('unexpected_keys:', res.unexpected_keys)
        # 形状不一致の検出
        msd = e2e_model.state_dict()
        shape_mismatch = []
        for k, v in mapped_state_dict.items():
            if k in msd and msd[k].shape != v.shape:
                shape_mismatch.append((k, tuple(msd[k].shape), tuple(v.shape)))
                print('shape_mismatch (first 20):', shape_mismatch[:20])

    except RuntimeError as e:
        print('load_state_dict RuntimeError:', e)
    e2e_model.load_state_dict(mapped_state_dict)
    e2e_model.eval()

    # Test the model
    sample_input = torch.rand(1, 15)
    pytorch_output = e2e_model(sample_input)

    # Convert to TFLite
    tfLite_model = ai_edge_torch.convert(e2e_model, (sample_input,))
    tfLite_model.export('gen_ppo.tflite')

    # Compare the outputs
    tflite_output = tfLite_model(sample_input)
    print("PyTorch output:", pytorch_output)
    print("TFLite output:", tflite_output)

if __name__ == "__main__":
    convert_network()
