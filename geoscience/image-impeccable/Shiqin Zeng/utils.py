

import torch
import numpy as np



def rescale_volume(seismic, low=0, high=100):
    """
    Rescaling 3D seismic volumes 0-255 range, clipping values between low and high percentiles
    """
    minval = np.percentile(seismic, low)
    maxval = np.percentile(seismic, high)
    seismic = np.clip(seismic, minval, maxval)
    seismic = ((seismic - minval) / (maxval - minval))*255

    return seismic
    
    
    
    
def onecube1(pre_index, steps, model, test_data, device,transpose=True):
    result_tensor = torch.empty(0).to(device)

    with torch.no_grad():
        if transpose:
            # Process the initial image slice
            images = test_data[0:200].transpose(0, 2, 1)
            images = images[np.newaxis, np.newaxis, :, :, :]
            images = torch.from_numpy(images).to(device)

            outputs = model(images).to(device)
            squeezed_tensor = outputs.squeeze()[0:pre_index, :, :]
            result_tensor = squeezed_tensor

            # Free memory
            del outputs, images
            torch.cuda.empty_cache()

            # Iterate through steps
            for idx in range(steps):
                images = test_data[idx * 200 + pre_index:(idx + 1) * 200 + pre_index].transpose(0, 2, 1)
                images = images[np.newaxis, np.newaxis, :, :, :]
                images = torch.from_numpy(images).to(device)

                outputs = model(images).to(device)
                squeezed_tensor = outputs.squeeze()
                result_tensor = torch.cat((result_tensor, squeezed_tensor), dim=0).to(device)

                # Free memory
                del outputs, images, squeezed_tensor
                torch.cuda.empty_cache()

            # Handle final slice case
            if result_tensor.shape[0] == 1259:
                return result_tensor.detach().cpu().numpy().transpose(0, 2, 1)
            else:
                images = test_data[1059:1259].transpose(0, 2, 1)
                images = images[np.newaxis, np.newaxis, :, :, :]
                images = torch.from_numpy(images).to(device)

                outputs = model(images).to(device)
                squeezed_tensor = outputs.squeeze()
                result_tensor = torch.cat((result_tensor, squeezed_tensor[pre_index - 59:200, :, :]), dim=0).to(device)

                # Free memory
                del outputs, images, squeezed_tensor
                torch.cuda.empty_cache()

                result = result_tensor.detach().cpu().numpy().transpose(0, 2, 1)
                return result

        else:
            # Non-transposed case
            images = test_data[0:200]
            images = images[np.newaxis, np.newaxis, :, :, :]
            images = torch.from_numpy(images).to(device)

            outputs = model(images).to(device)
            squeezed_tensor = outputs.squeeze()[0:pre_index, :, :]
            result_tensor = squeezed_tensor

            # Free memory
            del outputs, images
            torch.cuda.empty_cache()

            for idx in range(steps):
                images = test_data[idx * 200 + pre_index:(idx + 1) * 200 + pre_index]
                images = images[np.newaxis, np.newaxis, :, :, :]
                images = torch.from_numpy(images).to(device)

                outputs = model(images).to(device)
                squeezed_tensor = outputs.squeeze()
                result_tensor = torch.cat((result_tensor, squeezed_tensor), dim=0).to(device)

                # Free memory
                del outputs, images, squeezed_tensor
                torch.cuda.empty_cache()

            if result_tensor.shape[0] == 1259:
                return result_tensor.detach().cpu().numpy()
            else:
                images = test_data[1059:1259]
                images = images[np.newaxis, np.newaxis, :, :, :]
                images = torch.from_numpy(images).to(device)

                outputs = model(images).to(device)
                squeezed_tensor = outputs.squeeze()
                result_tensor = torch.cat((result_tensor, squeezed_tensor[pre_index - 59:200, :, :]), dim=0).to(device)

                # Free memory
                del outputs, images, squeezed_tensor
                torch.cuda.empty_cache()

                result = result_tensor.detach().cpu().numpy()
                return result
                
                
def onecube(pre_index, model, test_data,device, transpose=False, flip_vertical=False, flip_horizontal=False):
    result_tensor = torch.empty(0).to(device)
    steps = (len(test_data) - pre_index) // 200

    with torch.no_grad():
        # Transpose the data if needed
        if transpose:
            test_data = test_data.transpose(0, 2, 1)

        # Flip the data if needed
        if flip_vertical:
            test_data = np.flip(test_data, axis=1)

        if flip_horizontal:
            test_data = np.flip(test_data, axis=2)

         # Create a copy to remove any negative strides
        test_data = test_data.copy()
        # Create the first batch
        images = test_data[np.newaxis, np.newaxis, :, :, :]
        images = torch.from_numpy(images).to(device)
        outputs = model(images[:, :, 0:200, :, :]).to(device)
        squeezed_tensor = outputs.squeeze()[0:pre_index, :, :]
        result_tensor = squeezed_tensor.detach()  # Detach to release the computation graph

        # Free memory
        del outputs, test_data
        torch.cuda.empty_cache()

        # Iterate through the steps
        for idx in range(steps):
            input_data = images[:, :, idx * 200 + pre_index:(idx + 1) * 200 + pre_index, :, :]
            outputs = model(input_data).to(device)
            squeezed_tensor = outputs.squeeze()
            result_tensor = torch.cat((result_tensor, squeezed_tensor.detach()), dim=0).to(device)
            # Free memory
            del outputs, input_data, squeezed_tensor
            torch.cuda.empty_cache()

        # Handle the final slice case
        if result_tensor.shape[0] == 1259:
            result = result_tensor.detach().cpu().numpy()
        else:
            input_data = images[:, :, 1059:1259, :, :]
            outputs = model(input_data).to(device)
            squeezed_tensor = outputs.squeeze()
            result_tensor = torch.cat((result_tensor, squeezed_tensor[(200 + len(result_tensor) - 1259):200, :, :]), dim=0).to(device)

            # Free memory
            del outputs, input_data, squeezed_tensor
            torch.cuda.empty_cache()

            result = result_tensor.detach().cpu().numpy()

        # Apply transformations before returning the result


        if flip_horizontal:
            result = np.flip(result, axis=2)
        if flip_vertical:
            result = np.flip(result, axis=1)
        if transpose:
            result = result.transpose(0, 2, 1)
        return result



def segment_transfer(data, transpose_prob=0.5, flip_vertical_prob=0.5, flip_horizontal_prob=0.5):

    # Transpose the data if needed
    if torch.rand(1).item() < transpose_prob:
        data = data.permute(0, 1, 2, 4, 3)
        transpose = True
    else:
        transpose = False

    # Flip the data if needed
    if torch.rand(1).item() < flip_vertical_prob:
        data = torch.flip(data, dims=[3])  # Flip along the vertical axis (axis 3)
        flip_vertical = True
    else:
        flip_vertical = False

    if torch.rand(1).item() < flip_horizontal_prob:
        data = torch.flip(data, dims=[4])  # Flip along the horizontal axis (axis 4)
        flip_horizontal = True
    else:
        flip_horizontal = False

    return  data,transpose, flip_vertical, flip_horizontal

def segment_result_transfer(data, transpose, flip_vertical, flip_horizontal):

    if flip_horizontal:
        data = torch.flip(data, dims=[4])  # Unflip horizontal
    if flip_vertical:
        data = torch.flip(data, dims=[3])  # Unflip vertical
    if transpose:
        data = data.permute(0, 1, 2, 4, 3)  # Untranspose

    return data
                

def onecube_seg(pre_index, model, test_data, device,transpose=False, flip_vertical=False, flip_horizontal=False,seg= True):
    result_tensor = torch.empty(0).to(device)
    steps = (len(test_data) - pre_index) // 200

    with torch.no_grad():
        # Transpose the data if needed
        if transpose:
            test_data = test_data.transpose(0, 2, 1)

        # Flip the data if needed
        if flip_vertical:
            test_data = np.flip(test_data, axis=1)

        if flip_horizontal:
            test_data = np.flip(test_data, axis=2)

         # Create a copy to remove any negative strides
        test_data = test_data.copy()
        # Create the first batch
        images = test_data[np.newaxis, np.newaxis, :, :, :]
        images = torch.from_numpy(images).to(device)
        outputs = model(images[:, :, 0:200, :, :]).to(device)
        squeezed_tensor = outputs.squeeze()[0:pre_index, :, :]
        result_tensor = squeezed_tensor.detach()  # Detach to release the computation graph

        # Free memory
        del outputs, test_data
        torch.cuda.empty_cache()

        # Iterate through the steps
        for idx in range(steps):
            input_data = images[:, :, idx * 200 + pre_index:(idx + 1) * 200 + pre_index, :, :]
            if seg:
              input_data, seg_t,seg_v,seg_h = segment_transfer(input_data)

            outputs = model(input_data).to(device)
            if seg:
              outputs = segment_result_transfer(outputs,seg_t,seg_v,seg_h)
            squeezed_tensor = outputs.squeeze()
            result_tensor = torch.cat((result_tensor, squeezed_tensor.detach()), dim=0).to(device)

            # Free memory
            del outputs, input_data, squeezed_tensor
            torch.cuda.empty_cache()

        # Handle the final slice case
        if result_tensor.shape[0] == 1259:
            result = result_tensor.detach().cpu().numpy()
        else:
            input_data = images[:, :, 1059:1259, :, :]
            if seg:
              input_data, seg_t,seg_v,seg_h = segment_transfer(input_data)
            outputs = model(input_data).to(device)
            if seg:
              outputs = segment_result_transfer(outputs,seg_t,seg_v,seg_h)
            squeezed_tensor = outputs.squeeze()
            result_tensor = torch.cat((result_tensor, squeezed_tensor[(200 + len(result_tensor) - 1259):200, :, :]), dim=0).to(device)

            # Free memory
            del outputs, input_data, squeezed_tensor
            torch.cuda.empty_cache()

            result = result_tensor.detach().cpu().numpy()

        # Apply transformations before returning the result


        if flip_horizontal:
            result = np.flip(result, axis=2)
        if flip_vertical:
            result = np.flip(result, axis=1)
        if transpose:
            result = result.transpose(0, 2, 1)
        return result
    
def inference_3d(model, test_data,device):
    model.eval()
    result =0

    result5 =  onecube1(59, 5, model, test_data, device,transpose=False)
    result+=result5
    del result5
    torch.cuda.empty_cache()

    result6= onecube1(109, 5, model, test_data,device, transpose=True)
    result+=result6
    del result6
    torch.cuda.empty_cache()

    result7 = onecube1(159, 5, model, test_data, device,transpose=True)
    result+=result7
    del result7
    torch.cuda.empty_cache()

    result8 = onecube(40, model, test_data, device,transpose=False, flip_vertical=False, flip_horizontal=False)
    result+=result8
    del result8
    torch.cuda.empty_cache()

    result9 = onecube(50, model, test_data, device,transpose=True, flip_vertical=False, flip_horizontal=False)
    result+=result9
    del result9
    torch.cuda.empty_cache()

    result10 = onecube(120, model, test_data,device, transpose=True, flip_vertical=False, flip_horizontal=False)
    result+=result10
    del result10
    torch.cuda.empty_cache()

    result11 = onecube(109, model, test_data, device, transpose=True, flip_vertical=False, flip_horizontal=False)
    result+=result11
    del result11
    torch.cuda.empty_cache()


    result12 = onecube(159, model, test_data,device,  transpose=True, flip_vertical=False, flip_horizontal=False)
    result+=result12
    del result12
    torch.cuda.empty_cache()

    result13 = onecube(109, model, test_data,device,  transpose=True, flip_vertical=True, flip_horizontal=False)
    result+=result13
    del result13
    torch.cuda.empty_cache()


    result14 = onecube(159, model, test_data, device, transpose=True, flip_vertical=False, flip_horizontal=True)
    result+=result14
    del result14
    torch.cuda.empty_cache()

    result15 = onecube(169, model, test_data, device, transpose=False, flip_vertical=False, flip_horizontal=True)
    result+=result15
    del result15
    torch.cuda.empty_cache()

    result16 = onecube(179, model, test_data,device,  transpose=True, flip_vertical=True, flip_horizontal=False)
    result+=result16
    del result16
    torch.cuda.empty_cache()

    result = rescale_volume(result)
    result = result.astype(np.float16)
    print(result.shape)
    return result  
