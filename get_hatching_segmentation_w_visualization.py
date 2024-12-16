import os
import cv2
import numpy as np
import argparse
import tensorflow as tf
from tensorflow.keras.models import load_model

# Configure GPU usage
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # Match IDs with nvidia-smi
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Specify GPU 0


def crop_image(img, slice_size, stride):
    """
    Crop the input image into smaller slices.

    Parameters:
        img (numpy.ndarray): Input grayscale image.
        slice_size (int): Size of each slice.
        stride (int): Stride for cropping.

    Returns:
        list: Cropped image slices.
        int: Original image height.
        int: Original image width.
    """
    result_img = []
    height, width = img.shape

    for slice_w in range(int((width - slice_size) / stride) + 1):
        for slice_h in range(int((height - slice_size) / stride) + 1):
            upper = slice_h * stride
            lower = upper + slice_size
            left = slice_w * stride
            right = left + slice_size

            if lower > height:
                lower = height-1
            if right > width:
                right = width-1

            tmp_img = img[upper:lower, left:right]
            if tmp_img.shape == (slice_size, slice_size):
                result_img.append(tmp_img)

    return result_img, height, width


def load_img_as_cropped_set(image_path, slice_size, stride):
    """
    Load an image and crop it into slices.

    Parameters:
        image_path (str): Path to the input image.
        slice_size (int): Size of each slice.
        stride (int): Stride for cropping.

    Returns:
        numpy.ndarray: Cropped image slices.
        int: Original image height.
        int: Original image width.
        numpy.ndarray: Trimmed grayscale image.
        numpy.ndarray: Original colored image.
    """
    org_im = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    height, width = org_im.shape
    croph = stride * (height // stride)
    cropw = stride * (width // stride)
    trimmed_im = org_im[:croph, :cropw]
    padded_im = np.pad(trimmed_im, pad_width=slice_size - stride, mode='constant', constant_values=0)
    sliced_img, padded_h, padded_w= crop_image(padded_im, slice_size, stride)
    
    sliced_img = np.asarray(sliced_img).reshape(len(sliced_img), slice_size, slice_size, 1)
    colored_img = cv2.imread(image_path)[:croph, :cropw, :]

    return sliced_img, padded_h,padded_w, trimmed_im, colored_img


def rebuild_result_image_median(height, width, slice_size, stride, prob_set_max):
    """
    Rebuild the full image from predicted probabilities using a median filter.

    Parameters:
        height (int): Height of the padded image.
        width (int): Width of the padded image.
        slice_size (int): Size of each slice.
        stride (int): Stride used for cropping.
        prob_set_max (numpy.ndarray): Predicted probabilities for slices.

    Returns:
        numpy.ndarray: Reconstructed image.
        numpy.ndarray: Raw predictions.
    """
    depth = slice_size // stride
    raw_predictions_img = np.reshape(
        prob_set_max, (int((height - slice_size) / stride) + 1, int((width - slice_size) / stride) + 1), order='F'
    )

    prd_h, prd_w = raw_predictions_img.shape
    res_img = np.zeros((stride * (prd_h - depth + 1), stride * (prd_w - depth + 1)))

    for i in range(prd_w - depth + 1):
        for j in range(prd_h - depth + 1):
            res = np.median(raw_predictions_img[j:j + depth, i:i + depth])
            res_img[j * stride:(j + 1) * stride, i * stride:(i + 1) * stride] = res

    return res_img, raw_predictions_img


def visualize_results(org_img, colored_img, pred, vis_root, filename):
    """
    Visualize and save results as images with color mapping and overlays. 
    use a gamma function for visualization of output probabilities. 

    Parameters:
        org_img (numpy.ndarray): Original grayscale image.
        colored_img (numpy.ndarray): Original colored image.
        pred (numpy.ndarray): Predicted probabilities.
        vis_root (str): Directory to save visualizations.
        filename (str): Name of the output file.
    """
    huemax = 160
    saturationmax = 200
    h, w = org_img.shape

    # Create a color bar for predictions
    org_img_bar = 255 * np.ones((h, 110), dtype='uint8')
    org_img_bar[:, 100:110] = 0
    pred_bar = np.zeros(org_img_bar.shape, dtype=float)
    preds_bar = np.arange(0, 1.01, 0.05)
    
    x = int(h / 21)
    for i, p in enumerate(preds_bar):
        pred_bar[i * x:(i + 1) * x, :] = p

    rest = h - 21 * x
    pred_bar[21 * x:, :] = 0
    org_img_bar[21 * x:, :] = 0

    img_bar = cv2.cvtColor(org_img_bar, cv2.COLOR_GRAY2RGB)
    hsv_im_bar = cv2.cvtColor(img_bar, cv2.COLOR_RGB2HSV)
    hsv_im_bar[..., 0] = 35 + huemax * (
        np.sqrt(0.5 * np.ones(pred_bar.shape)) *
        np.sign(pred_bar - 0.5 * np.ones(pred_bar.shape)) *
        np.sqrt(np.abs(pred_bar - 0.5 * np.ones(pred_bar.shape))) +
        0.5 * np.ones(pred_bar.shape)
    )
    hsv_im_bar[..., 1] = saturationmax * np.ones(pred_bar.shape)
    rgb_imc = cv2.cvtColor(hsv_im_bar, cv2.COLOR_HSV2BGR)

    # Convert prediction to colored overlay
    img = cv2.cvtColor(org_img, cv2.COLOR_GRAY2RGB)
    hsv_im = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    hsv_im[..., 0] = 35 + huemax * (
        np.sqrt(0.5 * np.ones(pred.shape)) *
        np.sign(pred - 0.5 * np.ones(pred.shape)) *
        np.sqrt(np.abs(pred - 0.5 * np.ones(pred.shape))) +
        0.5 * np.ones(pred.shape)
    )
    hsv_im[..., 1] = saturationmax * np.ones(pred.shape)
    rgb_im = cv2.cvtColor(hsv_im, cv2.COLOR_HSV2BGR)

    # Save the results. Hatching segmentations will be shown as a heatmap overlaid the grayscale image
    os.makedirs(vis_root, exist_ok=True)
    output_path = os.path.join(vis_root, filename + '_hatchings_on_grayscale.png')
    cv2.imwrite(output_path, np.hstack((rgb_imc, rgb_im)))

    # hsv_im[..., 2] = 255 * np.ones(pred.shape)
    # rgb_im2 = cv2.cvtColor(hsv_im, cv2.COLOR_HSV2BGR)
    # output_path = os.path.join(vis_root, filename +'_whitebackground.png')
    # cv2.imwrite(output_path, rgb_im2)
    ### Segmented grayscale image and original color image side by side 
    output_path = os.path.join(vis_root, filename + '_segmented_grayscale_original_sidebyside.png')
    cv2.imwrite(output_path, np.hstack((rgb_imc, rgb_im, colored_img)))


def main():
    parser = argparse.ArgumentParser(description="Run image regression with pre-trained model.")
    parser.add_argument("--model", required=True, help="Path to the pre-trained model.")
    parser.add_argument("--image", required=True, help="Path to the input image.")
    parser.add_argument("--stride", type=int, default=1, help="Stride for cropping.")
    parser.add_argument("--slice_size", type=int, default=50, help="Size of each slice.")
    parser.add_argument("--output", required=True, help="Directory for saving results.")
    args = parser.parse_args()

    # Load pretrained detection model
    model = load_model(args.model)

    # Process image
    sliced_img, padded_height, padded_width, trimmed_img, colored_img = load_img_as_cropped_set(
        args.image, args.slice_size, args.stride
    )

    prob_set = model.predict(np.concatenate((sliced_img, sliced_img, sliced_img), axis =3))
    result, _ = rebuild_result_image_median(padded_height,padded_width, args.slice_size, args.stride, prob_set)

    # Visualize results
    filename = args.image.split('/')[-1].replace('.jpg','').replace('.png','')
    visualize_results(trimmed_img, colored_img, result, args.output, filename)
    print("Processing complete. Results saved to:", args.output)


if __name__ == "__main__":
    main()
