from utils import inference_3d
import time
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np

def prediction(noise_data,pre_filenames,device,model,save_3d_cube = False,visualize = True,save_path = "submission_files/predictions.npz"):

    device = device
    output_dict = {}
    for i in range(len(noise_data)):
        # Get the noisy data
        noise_data_pre = noise_data[i]

        # Start timing before the inference
        start_time = time.time()

        # Perform inference using the 3D model
        predicted_data = inference_3d(model, noise_data_pre,device)
        predicted_data = predicted_data.astype(np.float16)

        if save_3d_cube:
             save_path = pre_filenames[i] + '.npy'
             np.save(save_path, predicted_data)
             print(f"Predicted 3d data saved to {save_path}")


        # End timing after the inference
        end_time = time.time()

        # Calculate the time taken for prediction
        prediction_time = end_time - start_time
        print(f"Prediction {i+1} took {prediction_time:.4f} seconds")

        # Data keys for the images
        data_keys_x = [
            pre_filenames[i]+"_gt.npy-x_0",
            pre_filenames[i]+"_gt.npy-x_1",
            pre_filenames[i]+"_gt.npy-x_2"
        ]
        data_keys_i = [
            pre_filenames[i]+"_gt.npy-i_0",
            pre_filenames[i]+"_gt.npy-i_1",
            pre_filenames[i]+"_gt.npy-i_2"
        ]
        # Indices for predictions
        index = [75, 150, 225]

        #visualize the results, we only take 1 cube to visualize 3 sliced data
        if visualize and i == 0:

          # prediction
          for idx, key in enumerate(data_keys_i):
              output = predicted_data[:, :, index[idx]]
              output_dict[key] = output.T  # Store output with the specified key

              noise = noise_data[i][:,:,index[idx]]
              plt.figure(figsize=(15, 15))
              noise_vis = noise*23.740541458129883+223.07626342773438

              # Subplot 1: Noisy Data
              ax1 = plt.subplot(1, 3, 1)
              im1 = ax1.imshow(noise_vis, cmap="gray")
              ax1.set_title("Noisy Data")
              ax1.set_xlabel("X Axis")
              ax1.set_ylabel("Y Axis")
              divider1 = make_axes_locatable(ax1)
              cax1 = divider1.append_axes("right", size="5%", pad=0.05)
              plt.colorbar(im1, cax=cax1)

              # Subplot 2: Denoised Data
              ax2 = plt.subplot(1,3, 2)
              im2 = ax2.imshow(output, cmap="gray",vmin = 0,vmax=255)
              ax2.set_title("Denoised Data")
              ax2.set_xlabel("X Axis")
              ax2.set_ylabel("Y Axis")
              divider2 = make_axes_locatable(ax2)
              cax2 = divider2.append_axes("right", size="5%", pad=0.05)
              plt.colorbar(im2, cax=cax2)

              # Subplot 2: Denoised Data
              ax2 = plt.subplot(1,3, 3)
              im2 = ax2.imshow(noise_vis-output, cmap="gray")
              ax2.set_title("Predicted Noise")
              ax2.set_xlabel("X Axis")
              ax2.set_ylabel("Y Axis")
              divider2 = make_axes_locatable(ax2)
              cax2 = divider2.append_axes("right", size="5%", pad=0.05)
              plt.colorbar(im2, cax=cax2)

              plt.tight_layout()
              plt.show()

          # Loop over the keys and store corresponding outputs
        for idx, key in enumerate(data_keys_x):
            output = predicted_data[:, index[idx], :]
            output_dict[key] = output.T  # Store output with the specified key


    # Save the dictionary as an .npz file
    np.savez(save_path, **output_dict)
    print("Sliced data saved successfully.")


