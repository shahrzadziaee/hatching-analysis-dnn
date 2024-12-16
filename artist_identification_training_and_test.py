import os
import numpy as np
import h5py
import tensorflow as tf
import tensorflow.keras as keras
import matplotlib.pyplot as plt
from argparse import ArgumentParser

# Set environment variables for GPU usage
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '1'

# Dictionary to map model names to corresponding Keras applications models
AVAILABLE_MODELS = {
    "VGG16": keras.applications.vgg16,
    "EfficientNetB0": keras.applications.efficientnet,
    "ResNet50": keras.applications.resnet,
    "MobileNet": keras.applications.mobilenet,
    "DenseNet121": keras.applications.densenet
}

BASE_LAYER_NAME= {
    "DenseNet121": 'densenet121',
    "MobileNet": 'mobilenet_1.00_224',
    "VGG16": "vgg16",
    "EfficientNetB0": 'efficientnetb0',
    "ResNet50": 'resnet50'
}

def parse_args():
    """
    Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = ArgumentParser(description="Train an artist identification model using a pre-trained base model.")
    parser.add_argument('--train_data_paths', type=str, nargs='+', required=True, help="List of paths to the training HDF5 datasets")
    parser.add_argument('--val_data_paths', type=str, nargs='+', required=True, help="List of paths to the validation HDF5 datasets")
    parser.add_argument('--class_labels', type=int, nargs='+', help="List of class labels corresponding to the datasets")
    parser.add_argument('--output_root', type=str, required=True, help="Path to the output directory")
    parser.add_argument('--checkpoint', type=str, required=True, help="Path to the pre-trained detection model checkpoint for transfer learning")
    parser.add_argument('--base_model', type=str, default='MobileNet', choices=AVAILABLE_MODELS.keys(),
                        help="Choose the base model (DenseNet121, VGG16, EfficientNetB0, ResNet50, MobileNet)")
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size (default: 32)")
    parser.add_argument('--epochs', type=int, default=100, help="Number of epochs (default: 100)")
    parser.add_argument('--learning_rate', type=float, default=1e-7, help="Learning rate (default: 1e-7)")
    parser.add_argument('--number_of_artists', type=int, default=None, help="Number of artist classes (optional)")
    parser.add_argument('--seed', type=int, default=42, help="Random seed for reproducibility")
    return parser.parse_args()

def load_balance_datasets(dataset_paths, class_labels, seed=42):
    """
    Load and balance multiple datasets.

    Args:
        dataset_paths (list of str): Paths to the datasets.
        class_labels (list of int): Corresponding class labels.
        seed (int): Random seed for reproducibility.

    Returns:
        Tuple: Balanced images and labels.
    """
    if len(dataset_paths) != len(class_labels):
        raise ValueError("The number of datasets and class labels must match.")
    if not all(-1< class_label < len(class_labels) for class_label in class_labels):
        raise ValueError(f"The class labels must be in range [0,{len(class_labels)})")


    # Determine the minimum size among all datasets
    min_size = float('inf')
    for dataset_path in dataset_paths:
        with h5py.File(dataset_path, 'r') as f:
            min_size = min(min_size, len(f['images']))

    # Load and balance datasets
    np.random.seed(seed)
    balanced_images = []
    balanced_labels = []

    for dataset_path, class_label in zip(dataset_paths, class_labels):
        with h5py.File(dataset_path, 'r') as f:
            images = f['images'][:]
            labels = np.full(len(images), class_label)

            # Shuffle and sample
            indices = np.random.choice(len(images), min_size, replace=False)
            balanced_images.append(images[indices])
            balanced_labels.append(labels[indices])

    # Combine and shuffle
    balanced_images = np.vstack(balanced_images)
    balanced_labels = np.concatenate(balanced_labels)
    shuffle_indices = np.random.permutation(len(balanced_images))
    balanced_images = balanced_images[shuffle_indices]
    balanced_labels = balanced_labels[shuffle_indices]

    return balanced_images, balanced_labels

def create_model(base_model_name, checkpoint_path, number_of_artists):
    """
    Create a Keras model for artist identification using a pre-trained detection base model.

    Args:
        base_model_name (str): Name of the base model (e.g., 'DenseNet121').
        checkpoint_path (str): Path to the pre-trained model checkpoint.
        number_of_artists (int): Number of artist classes.

    Returns:
        Keras model
    """
    base_model = keras.models.load_model(checkpoint_path)
    base_input = base_model.get_layer(BASE_LAYER_NAME[base_model_name]).input
    base_output = base_model.get_layer(BASE_LAYER_NAME[base_model_name]).output

    base = keras.Model(base_input, base_output)
    base.trainable = True

    inputs = keras.layers.Input([50, 50, 3], dtype=tf.float32)
    x = AVAILABLE_MODELS[base_model_name].preprocess_input(inputs)
    x = base(x, training=False)
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dropout(0.5)(x)

    outputs = keras.layers.Dense(number_of_artists, activation='softmax')(x)

    return keras.Model(inputs, outputs)

def plot_training_results(history, plots_path):
    """
    Plot training and validation accuracy and loss.

    Args:
        history: Training history object.
        plots_path (str): Path to save the plots.
    """
    acc_train = history.history['categorical_accuracy']
    acc_val = history.history['val_categorical_accuracy']
    loss_train = history.history['loss']
    loss_val = history.history['val_loss']

    plt.figure(figsize=(8, 8))
    plt.subplot(2, 1, 1)
    plt.plot(acc_train, label='Training Accuracy')
    plt.plot(acc_val, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.ylabel('Accuracy')
    plt.grid()
    plt.title('Training and Validation Accuracy')

    plt.subplot(2, 1, 2)
    plt.plot(loss_train, label='Training Loss')
    plt.plot(loss_val, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.ylabel('Loss')
    plt.grid()
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')

    plt.savefig(plots_path)
    plt.close()

def main():
    # Parse command-line arguments
    args = parse_args()

    # Paths and hyperparameters
    train_dataset_paths = args.train_data_paths
    val_dataset_paths = args.val_data_paths
    class_labels = args.class_labels if args.class_labels else list(range(len(train_dataset_paths)))
    output_root = args.output_root
    checkpoint_path = args.checkpoint
    base_model_name = args.base_model
    batch_size = args.batch_size
    epochs = args.epochs
    learning_rate = args.learning_rate
    number_of_artists = len(class_labels) if class_labels else args.number_of_artists
    seed = args.seed

    logs_root = os.path.join(output_root, 'logs')
    plots_root = os.path.join(output_root, 'plots')
    snaps_root = os.path.join(output_root, 'snapshots')

    os.makedirs(logs_root, exist_ok=True)
    os.makedirs(plots_root, exist_ok=True)
    os.makedirs(snaps_root, exist_ok=True)

    # Load and balance datasets
    x_train, z_train = load_balance_datasets(train_dataset_paths, class_labels, seed)
    x_val, z_val = load_balance_datasets(val_dataset_paths, class_labels, seed)

    x_train = np.concatenate((x_train,x_train,x_train), axis=-1)
    x_val = np.concatenate((x_val,x_val,x_val), axis=-1)

    # Create model
    artist_identification_model = create_model(base_model_name, checkpoint_path, number_of_artists)
    artist_identification_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[keras.metrics.SparseCategoricalAccuracy(name='categorical_accuracy')]
    )

    # Define callbacks
    checkpoint_dir = os.path.join(snaps_root, f'Artist_identification_{base_model_name}_{number_of_artists}classes_batchsize{batch_size}_lr{learning_rate}_epochs{epochs}_checkpoints/')
    os.makedirs(checkpoint_dir, exist_ok=True)

    checkpoint_callback = keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(checkpoint_dir, 'checkpoint'),
        save_weights_only=True,
        save_freq='epoch'
    )
    csv_logger = keras.callbacks.CSVLogger(os.path.join(logs_root, 'training_log.csv'))

    # Train model
    history = artist_identification_model.fit(
        x_train, z_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(x_val, z_val),
        callbacks=[checkpoint_callback, csv_logger]
    )

    # Save model
    artist_identification_model.save(os.path.join(snaps_root, f'Artist_identification_{base_model_name}_{number_of_artists}classes_batchsize{batch_size}_lr{learning_rate}_epochs{epochs}.h5'))

    # Plot results
    plot_training_results(history, os.path.join(plots_root, f'training_results_{base_model_name}_{number_of_artists}classes.png'))

if __name__ == '__main__':
    main()
