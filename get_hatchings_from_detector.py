import os
import numpy as np
import h5py
import cv2
import tensorflow as tf
import tensorflow.keras as keras
from argparse import ArgumentParser

# Set default GPU options
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # Set GPU device to use

def parse_args():
    """ Parse command-line arguments """
    parser = ArgumentParser(description="Process images for regression")
    parser.add_argument('--model', type=str, required=True, help="Path to the pre-trained detection model")
    parser.add_argument('--image_dir', type=str, required=True, help="Path to the image directory")
    parser.add_argument('--output_dir', type=str, required=True, help="Path to save the HDF5 file")
    return parser.parse_args()

def crop_image(img, slice_size, stride):
    """ Crop image into smaller patches """
    result_img = []
    height, width = img.shape

    for slice_w in range((width - slice_size) // stride + 1):
        for slice_h in range((height - slice_size) // stride + 1):
            upper, lower = slice_h * stride, slice_h * stride + slice_size
            left, right = slice_w * stride, slice_w * stride + slice_size

            if lower > height:
                lower = height
            if right > width:
                right = width

            tmp_img = img[upper:lower, left:right]
            if tmp_img.shape[0] > 0 and tmp_img.shape[1] > 0:
                result_img.append(tmp_img)

    return np.asarray(result_img), height, width

def load_img_as_cropped_set(image_path, slice_size, stride):
    """ Load and crop image into smaller pieces """
    img = cv2.imread(image_path, 0)
    sliced_img, height, width = crop_image(img, slice_size, stride)
    return np.reshape(sliced_img, (len(sliced_img), slice_size, slice_size, 1)), height, width

def get_regression_on_cropped_set(model, sliced_img):
    """ Predict using the cropped image set """
    input_data = np.concatenate((sliced_img, sliced_img, sliced_img), axis=3)  # Convert to 3 channels
    return np.asarray(model.predict(input_data))

def process_predictions(predictions, threshold=0.5):
    """
    Process the predictions to filter out only those that meet the condition.
    We extract only predictions >= threshold (0.5 by default).
    """
    return np.where(predictions >= threshold)[0]  # Return indices of valid predictions

def process_image(model, image_path, slice_size, stride):
    """ Process image, get predictions, and filter them based on threshold """
    sliced_img, _, _ = load_img_as_cropped_set(image_path, slice_size, stride)
    predictions = get_regression_on_cropped_set(model, sliced_img)
    
    # Filter the valid predictions based on the threshold
    valid_indices = process_predictions(predictions)
    
    # Get valid cropped images and corresponding predictions
    valid_cropped_imgs = sliced_img[valid_indices, :]
    valid_predictions = predictions[valid_indices]
    
    return valid_cropped_imgs, valid_predictions

def main():
    # Parse command-line arguments
    args = parse_args()

    # Load pre-trained model
    model = keras.models.load_model(args.model)
    
    # Define metrics and compile model
    # metrics = [
    #     keras.metrics.TruePositives(name='tp'),
    #     keras.metrics.FalsePositives(name='fp'),
    #     keras.metrics.TrueNegatives(name='tn'),
    #     keras.metrics.FalseNegatives(name='fn'), 
    #     keras.metrics.BinaryAccuracy(name='accuracy'),
    #     keras.metrics.Precision(name='precision'),
    #     keras.metrics.Recall(name='recall'),
    #     keras.metrics.AUC(name='auc'),
    # ]
    
    # model.compile(optimizer=tf.keras.optimizers.Adam(lr=args.learning_rate),
    #               loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
    #               metrics=metrics)

    # Set parameters for cropping
    stride = 10
    slice_size = 50

    # Initialize lists to accumulate all image patches and predictions
    all_cropped_imgs = []
    all_predictions = []

    # Process each image in the image directory
    for img_file in os.listdir(args.image_dir):
        if img_file.endswith('.jpg') or img_file.endswith('.png'):
            image_path = os.path.join(args.image_dir, img_file)
            print(f"Processing: {image_path}")

            # Predict hatching regions and apply threshold filter
            cropped_imgs, predictions = process_image(model, image_path, slice_size, stride)
            
            # Append the processed images and predictions to the lists
            all_cropped_imgs.append(cropped_imgs)
            all_predictions.append(predictions)
    
    # Convert the lists to arrays
    all_cropped_imgs = np.vstack(all_cropped_imgs)  # Stack all patches into one array
    all_predictions = np.concatenate(all_predictions)  # Concatenate all predictions into one array
    
    # Shuffle the data
    shuffle_indices = np.random.permutation(len(all_cropped_imgs))
    all_cropped_imgs = all_cropped_imgs[shuffle_indices]
    all_predictions = all_predictions[shuffle_indices]

    # Save the shuffled data to HDF5
    with h5py.File(args.output_dir+'detected_hatchings.h5', 'w') as hdf5_file:
        hdf5_file.create_dataset('images', data=all_cropped_imgs)
        hdf5_file.create_dataset('predictions', data=all_predictions)
    
    print(f"Processing complete. Shuffled data saved to {args.output_dir}")

if __name__ == "__main__":
    main()
