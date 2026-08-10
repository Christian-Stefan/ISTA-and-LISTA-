from CodeForStudents4.imports_submission import *
from CodeForStudents4.utils_submission import *

class ProxNet(nn.Module):
    def __init__(self, conv_net: nn.Module, num_iterations: int = 5):
        super().__init__()
        # 1. The Proximal Operator P_psi/ConvISTA instance
        self.prox_net = conv_net
        self.num_iterations:int = num_iterations # Hard-coded 
        
        # 2. Learnable step sizes (mu) for each iteration. 
        # Initialized to 0.5. We use a Parameter to track gradients.
        self.mus = nn.Parameter(torch.full((num_iterations,), 0.5))

    def forward(self, kspace: torch.Tensor, M: torch.Tensor):
        # 3.1 Initial zero-filled reconstruction (Phi^H y) mimicking z_0 = x_t = 0
        x0_complex = get_accelerated_MRI(kspace * M)
        # 3.2 Initialize x_t as the real-valued magnitude image for the ConvNet
        x_t = torch.abs(x0_complex).to(torch.float32)

        for t in range(self.num_iterations):
            # 3.3 Enforce the constraint: mu must be between 0 and 1
            mu = torch.clamp(self.mus[t], 0.0, 1.0)

            # =================@@@==================
            # 4: Data Consistency (The Physics 'g')
            # =================@@@==================
            # 4.0 Extract and save the original phase to keep the physics intact
            currentphase = torch.angle(x0_complex) if t==0 else torch.angle(x_dc_complex)
            # 4.1. Multiply the real image by the phase to restore complex numbers
            x_t_complex = x_t.squeeze(1) * torch.exp(1j * currentphase)
            # 4.2. Forward physics: Image domain to k-space (Phi)
            k_t = get_k_space(x_t_complex)
            # 4.3. Gradient step in k-space: k_t - mu * M * (k_t - y)
            k_dc = k_t - mu * M * (k_t - kspace * M)
            # 4.4 Inverse physics: k-space back to Image domain (Phi^H)
            x_dc_complex = get_accelerated_MRI(k_dc)

            # ==================2@@=====================
            # 5.Prox
            # ==================2@@=====================
            # 5.1. Extract the absolute magnitude for the network
            x_dc_abs = torch.abs(x_dc_complex).to(torch.float32)
            
            # 5.2. Pass through the CNN to denoise/inpaint
            x_t = self.prox_net(x_dc_abs)

        return x_t, x0_complex