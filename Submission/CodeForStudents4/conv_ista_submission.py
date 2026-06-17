from CodeForStudents4.imports_submission import *

class ConvISTA(nn.Module):
    """
    Requirements: As a starting point for a deep learning solution, design an end-to-end convolutional neural network. 
    As input it should take an initial reconstruction from a partial k-space measurement and output a guess for the 
    final reconstruction. The net is not (yet) allowed to make use of knowledge about the sampling mask.
    """
    def __init__(self, pading:str|int, stride:int, kernel_size:int|tuple, width:list|int, layers:int = 4):
        super().__init__()
        # 1. Listing out the variables used to define the network's architecture
        self.NETWORK = nn.Sequential()
        self.width = width
        self.layers = layers

        self.pading = pading
        if type(self.pading) != list:
            self.pading = [pading for item in range(layers)]
        self.stride = stride

        self.ken_size = kernel_size
        if type(self.ken_size) != list:
            self.ken_size = [kernel_size for item in range(layers)]


        for layer in range(layers):
            self.NETWORK.append(nn.Conv2d(in_channels=width[layer], 
                                          out_channels=width[layer+1], 
                                          kernel_size=self.ken_size[layer],
                                          padding=self.pading[layer]))
            if layer != layers-1:
                self.NETWORK.append(nn.Tanh())
            else: 
                self.NETWORK.append(nn.ReLU())


    def forward(self, x):
        # Convert absolute value to standard float32 to preserve training velocity
        x_abs = torch.abs(x).to(torch.float32)
        x_final_rec = self.NETWORK(x_abs.unsqueeze(1))
        return x_final_rec