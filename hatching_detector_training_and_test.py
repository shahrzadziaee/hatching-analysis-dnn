import numpy as np
import h5py
import tensorflow as tf
import argparse
import os
import tensorflow.keras as keras
import matplotlib.pyplot as plt

# Set GPU visibility (optional)
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

# Dictionary to map model names to corresponding Keras applications models
AVAILABLE_MODELS = {
    "VGG16": keras.applications.vgg16,
    "EfficientNetB0": keras.applications.efficientnet,
    "ResNet50": keras.applications.resnet,
    "MobileNet": keras.applications.mobilenet,
    "DenseNet121": keras.applications.densenet
}
MODEL_CLASS= {
    "VGG16": keras.applications.VGG16,
    "EfficientNetB0": keras.applications.EfficientNetB0,
    "ResNet50": keras.applications.ResNet50,
    "MobileNet": keras.applications.MobileNet,
    "DenseNet121": keras.applications.DenseNet121
}

def parse_arguments():
    parser = argparse.ArgumentParser(description="Train a hatching detection model with a user-specified base model")
    parser.add_argument('--dataset_dir', type=str, required=True, help="Directory containing training and test datasets")
    parser.add_argument('--train_file', type=str, required=True, help="HDF5 file for training data")
    parser.add_argument('--test_file', type=str, required=True, help="HDF5 file for test data")
    parser.add_argument('--epochs', type=int, default=100, help="Number of training epochs")
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size")
    parser.add_argument('--learning_rate', type=float, default=1e-7, help="Learning rate")
    parser.add_argument('--base_model', type=str, default='EfficientNetB0', choices=AVAILABLE_MODELS.keys(),
                        help="Choose the base model (DenseNet121, VGG16, EfficientNetB0, ResNet50, MobileNet)")
    parser.add_argument('--output_dir', type=str, required=True, help="Path to the output directory")
    return parser.parse_args()

def prepare_directories(outputdir, log_dir, snap_dir):
    for path in [log_dir, snap_dir]:
        if not os.path.isdir(os.path.join(outputdir,path)):
            os.makedirs(os.path.join(outputdir,path))
    return log_dir, snap_dir

def load_data(dataset_dir, train_file, test_file): 
    """
    Load train and test datasets
        Args:
        dataset_dir (list of str): Directory containing training and test data.
        train_file (str): HDF5 file for training dataset
        test_file (str): HDF5 file for test dataset

    Returns:
        Tuple: Arrays containing training and test data and hatching labels
    """
    # Training data
    with h5py.File(os.path.join(dataset_dir, train_file), 'r') as train_data:
        x_train = np.concatenate([np.reshape(train_data['images'], (len(train_data['images']), 50, 50, 1))] * 3, axis=3)
        y_train = np.asarray(train_data['hatching_labels'])
    
    # Test data
    with h5py.File(os.path.join(dataset_dir, test_file), 'r') as test_data:
        x_test = np.concatenate([np.reshape(test_data['images'], (len(test_data['images']), 50, 50, 1))] * 3, axis=3)
        y_test = np.asarray(test_data['hatching_labels'])
    
    return (x_train, y_train), (x_test, y_test)

def build_model(base_model_name, learning_rate, input_shape=(50, 50, 3)):
    # Select base model architecture, load imagenet pre-trained 
    base_model_class = MODEL_CLASS[base_model_name]
    base_model = base_model_class(weights='imagenet', input_shape=input_shape, include_top=False)
    base_model.trainable = True

    inputs = keras.Input(shape=input_shape, dtype=tf.float32)
    x = AVAILABLE_MODELS[base_model_name].preprocess_input(inputs) # Adjust preprocessing as needed for different models
    x = base_model(x, training=False)
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dropout(0.5)(x)
    outputs = keras.layers.Dense(1, activation='sigmoid')(x)
    
    model = keras.Model(inputs, outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss=keras.losses.BinaryCrossentropy(),
        metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall(), keras.metrics.AUC()]
    )
    return model

def train_model(model, x_train, y_train, x_test, y_test, log_dir, snap_dir, epochs, batch_size, learning_rate):
    checkpoint_path = os.path.join(snap_dir, 'model_checkpoint/')
    if not os.path.isdir(checkpoint_path):
        os.mkdir(checkpoint_path)

    checkpoint_callback = keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path + 'checkpoint',
        save_weights_only=True,
        save_freq='epoch'
    )
    
    csv_logger = tf.keras.callbacks.CSVLogger(os.path.join(log_dir, 'training_log.csv'))
    
    history = model.fit(
        x_train, y_train, 
        validation_data=(x_test, y_test),
        epochs=epochs, 
        batch_size=batch_size, 
        callbacks=[checkpoint_callback, csv_logger]
    )

    model.save(os.path.join(snap_dir, 'hatching_detection_trained_model.h5'))
    return history

def plot_history(history, log_dir):
    metrics = ['accuracy', 'loss', 'precision', 'recall']
    plt.figure(figsize=(8, 8))

    for metric in metrics:
        train_values = history.history[metric]
        val_values = history.history[f'val_{metric}']
        
        plt.plot(train_values, label=f'Training {metric.capitalize()}')
        plt.plot(val_values, label=f'Test {metric.capitalize()}')
        plt.title(f'Training and Test {metric.capitalize()}')
        plt.ylabel(metric.capitalize())
        plt.xlabel('Epoch')
        plt.legend()
        plt.grid()
        plt.savefig(os.path.join(log_dir, f'{metric}_curve.png'))
        plt.clf()

def main():
    args = parse_arguments()
    
    log_dir, snap_dir = prepare_directories(args.output_dir, 'logs', 'snapshots')
    
    (x_train, y_train), (x_test, y_test) = load_data(args.dataset_dir, args.train_file, args.test_file)
    
    model = build_model(args.base_model, args.learning_rate)
    history = train_model(model, x_train, y_train, x_test, y_test, log_dir, snap_dir, args.epochs, args.batch_size,args.learning_rate)
    
    plot_history(history, log_dir)

if __name__ == "__main__":
    main()
