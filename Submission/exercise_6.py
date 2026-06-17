from CodeForStudents4.imports_submission import *
from CodeForStudents4.utils_submission import *
from CodeForStudents4.conv_ista_submission import *
from CodeForStudents4.proxi_net_submission import *

def training(BATCH_SIZE,STRIDE,KERNEL_SIZE,WIDTH,PROXI_LR_NET,PROXI_LR_MUS,EPOCHS):
    # 1. Network Initialization and Setup definition
        # 1.1 Load the data
        train_MRI, test_MRI = create_dataloaders_MRI(batch_size=BATCH_SIZE-32, data_loc=r'CodeForStudents4\Fast_MRI_Knee')
        # 1.2 Check if Intel GPU acceleration is available
        if hasattr(torch, "xpu") and torch.xpu.is_available():
            device = torch.device("xpu") 
        elif hasattr(torch, 'cuda') and torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device("cpu")
        print("Device", device)

        # 1.3 Define both, ConvISTA network and ProxNet
        base_convnet = ConvISTA( pading='same',stride=STRIDE, kernel_size=KERNEL_SIZE, width=WIDTH)
        model = ProxNet( conv_net=base_convnet, num_iterations=5).to(device)
        # 1.4 Define the optimizer and augment the search by using weight decay and scheduler;
        optim = Adam([
            {'params': model.prox_net.parameters(), 'lr': PROXI_LR_NET}, 
            {'params': model.mus, 'lr': PROXI_LR_MUS} 
        ])
        loss_fn = MSELoss()
        # ... Scheduler configurations
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim,mode="min",factor=0.5,patience=2,min_lr=1e-6)
        # 1.5 Experiment configuration and creation of history holders
        epochs:int = EPOCHS-5
        train_loss:list = []
        val_loss:list = []
        best_val_loss = float('inf') # Initialize to infinity so the first epoch always saves
        os.makedirs(r'CodeForStudents4\Results\checkpoints', exist_ok=True) # Create a directory for weights;

        for epoch in range(epochs):
            # 2---------###-------##\
            # 2.Model Training Phase###\
            # 2---------###-----------##\
            model.train()
            train_loss_counter:float = 0.0
            
            for idx, (kspace, M, gt) in enumerate(tqdm.tqdm(train_MRI, desc=f"Train epoch {epoch}")):
                # 2.1. Push data to GPU
                kspace = kspace.to(device) # complete/raw kspace
                M = M.to(device) # MASK
                gt = gt.to(device) # ground truth image
                # 2.2 Zero out the gradients residuals possibly located in the cache memory 
                optim.zero_grad()
                # 2.3 Forward pass
                _train_rec = model(kspace, M)
                # 2.4 Calculate Loss (... unsqueeze gt to match channel dims)
                _train_loss, _ = loss_fn(_train_rec, gt.unsqueeze(1))
                # 2.5 Backprop & Step
                _train_loss.backward()
                optim.step()
                # 2.6 Tracking loss, determine its average over batches and append history
                train_loss_counter += _train_loss.item()
            avg_train_loss = train_loss_counter / len(train_MRI)
            train_loss.append(avg_train_loss)

            # 3---------###-------####\
            # 3.Model Training Phase###\
            # All the aforementioned ###\
            # steps will be repeated ####\
            # in this loop           #####\      
            # 3---------###-----------#####\
            model.eval()
            val_loss_counter:float = 0.0
            with torch.no_grad():
                for idx, (kspace, M, gt) in enumerate(tqdm.tqdm(test_MRI, desc=f"Test epoch {epoch}")):
                    kspace = kspace.to(device)
                    M = M.to(device)
                    gt = gt.to(device)
                    _test_rec, _ = model(kspace, M)
                    _test_loss = loss_fn(_test_rec, gt.unsqueeze(1))
                    val_loss_counter += _test_loss.item()

            avg_val_loss = val_loss_counter / len(test_MRI)
            val_loss.append(avg_val_loss)
            scheduler.step(avg_val_loss)
            current_lrs = [group["lr"] for group in optim.param_groups]
            print(f"Learning rates: prox_net={current_lrs[0]:.2e}, mus={current_lrs[1]:.2e}")
            print("mus:", model.mus.detach().cpu().numpy())

            # 4. Display results
            print(f"RESULTS Epoch {epoch}: Train Loss {avg_train_loss:.6f} | Test Loss {avg_val_loss:.6f}")
            # 5---------###-------##\
            # 5.Checkpoint Saving Logic###\
            # 5---------###-----------##\
            # 5.1 Always save the newest state at the end of the epoch
            torch.save(model.state_dict(), 'CodeForStudents4/Results/checkpoints/proxnet_latest.pth')

            # 5.2 Check if this is the best validation loss we have seen so far
            if avg_val_loss < best_val_loss:
                print(f"--> Validation loss improved from {best_val_loss:.6f} to {avg_val_loss:.6f}. Saving best model!")
                best_val_loss = avg_val_loss
                # Save the best model
                torch.save(model.state_dict(), 'CodeForStudents4/Results/checkpoints/proxnet_best.pth')
        
        
        PLT.figure(figsize=(8, 5))

        PLT.plot(train_loss, label="Training loss", marker="o")
        PLT.plot(val_loss, label="Test loss", marker="o")

        PLT.xlabel("Epoch")
        PLT.ylabel("Loss")
        PLT.title("Training and Test Loss Across Epochs")
        PLT.legend()
        PLT.grid(True)
        PLT.tight_layout()
        PLT.savefig("CodeForStudents4\Results\Training_Test_Loss.jpg")
        PLT.show()

        return model

def display_results(model, path, MODE, DATA)->None:
    # 1. Load the weighs stored at location indicated by path 
    test_MRI = DATA
    weights = torch.load(path, weights_only=True) # LOADING weights
    model.load_state_dict(weights) # INJECTING weights
    ###### === ########
    # Evaluation Mode #
    ###### === ########
    model.eval() 
    with torch.no_grad():
        for idx, (kspace, M, gt) in enumerate(tqdm.tqdm(test_MRI)):
            if MODE == 'ProxiNet':
                final_output, partial_output = model(kspace, M)
            else:
                pass

        partial_rec = partial_output.squeeze()
        final_rec = final_output.squeeze()
        clean=gt.squeeze()


        # 3. Displaying required results
        fig, axes = PLT.subplots(nrows=3, ncols=10, figsize=(25, 6))
        for i in range(10):
            # ROW 1: Partial Rec 
            axes[0, i].imshow(torch.abs(partial_rec[i]), cmap='gray', vmin=-1, vmax=1)
            axes[0, i].axis('off')
            if i == 0: axes[0, i].set_title("Partial reconstructions", fontsize=10, fontweight='bold', pad=10)
            
            # ROW 2: Network Reconstruction (x_K)
            axes[1, i].imshow(final_rec[i], cmap='gray', vmin=-1, vmax=1)
            axes[1, i].axis('off')
            if i == 0: axes[1, i].set_title(f"{MODE} Output", fontsize=10, fontweight='bold', pad=10)
            
            # ROW 3: Ground Truth Target (x)
            axes[2, i].imshow(clean[i], cmap='gray', vmin=-1, vmax=1)
            axes[2, i].axis('off')
            if i == 0: axes[2, i].set_title("Clean Truth", fontsize=10, fontweight='bold', pad=10)

    PLT.tight_layout()
    PLT.savefig(r'CodeForStudents4\Results\ConvNetOutput.jpg')
    PLT.show()



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Data panel, holding enviornment variables, experimental settings and some other parameters")
    parser.add_argument("--load",action="store_true", help="Load saved results and generate figures without retraining.")
    parser.add_argument("--Proxi_LR", type=dict, default={'prox_net':1e-4, 'mus':1e-2}, help="Dual-learning rateo optimizer, one dedicated to prox_net parameters and the other dedicated to mu")
    parser.add_argument("--ConvISTA_ARCH", type=ast.literal_eval, default=[1, 3, [1, 4, 8, 4, 1]], help="List entailing integers (, the first one representing the stride and the second the kernel size) and a list standing out for the width ConvISTA should adopt")
    parser.add_argument("--EPOCHS", type=int, default=10, help="Number of epochs (not fixed)")
    parser.add_argument("--BATCH_SIZE", type=int, default=64, help="Number of batches (not fixed)")
    parser.add_argument("--result_param_path", type=str, default='CodeForStudents4\Results\checkpoints\proxnet_best.pth',help="Path leading to the best parameters of ProxiNet")
    arg = parser.parse_args()

    _, test_MRI = create_dataloaders_MRI(batch_size=arg.BATCH_SIZE-32, data_loc=r'CodeForStudents4\Fast_MRI_Knee')

    if arg.load:
        print("================================= Disclaimer ============================")
        print("Python script .py is followed by argument --load therefore all the results will be generated based on existing/pre-loaded weights\n" \
        "All the weights were stored beforehand in CodeForStudents4\Results")
        print("====================== Loading Exercise 4.6.a) =================================")
        #TODO Written answer

        print("====================== Loading Exercise 4.6.b) =================================")
        img = image.open("CodeForStudents4\Results\Training_Test_Loss.jpg")
        img.show()
        

        print("====================== Loading Exercise 4.6.c) =================================")
        base_convnet = ConvISTA( pading='same',stride=arg.ConvISTA_ARCH[0], kernel_size=arg.ConvISTA_ARCH[1], width=arg.ConvISTA_ARCH[2])
        model = ProxNet(conv_net=base_convnet, num_iterations=5)
        display_results(model = model, path=arg.result_param_path, MODE='ProxiNet', DATA=test_MRI)

        print("====================== Loading Exercise 4.6.d) =================================")



        print("====================== Loading Exercise 4.6.e) =================================")

    else:
        print("==================DISCLAIMER==================\n"\
        "Python script .py is NOT followed by argument --load therefore all the results will be generated based on live model training that might be time consuming\n" \
        "After training, all the weights will be stored in CodeForStudents4\Results")

        print("====================== Loading Exercise 4.6.a) =================================")
        #TODO - Written answer

        print("====================== Loading Exercise 4.6.b) =================================")
        model = training(arg.BATCH_SIZE,arg.ConvISTA_ARCH[0],arg.ConvISTA_ARCH[1],arg.ConvISTA_ARCH[2],arg.Proxi_LR['prox_net'],arg.Proxi_LR['mus'],arg.EPOCHS)
        # Incorporate training-testing plot in training-loop
        print("====================== Loading Exercise 4.6.c) =================================")
        display_results(model = model, path=arg.result_param_path, MODE='ProxiNet', DATA=test_MRI)

        print("====================== Loading Exercise 4.6.d) =================================")



        print("====================== Loading Exercise 4.6.e) =================================")

