from keras.preprocessing.image import ImageDataGenerator
import numpy as np

def extract_features(directory, sample_count, batch_size=32):
    """Extracts features from images in a directory.

    Args:
        directory: The directory containing the images.
        sample_count: The total number of samples to extract.
        batch_size: The number of images to process in each batch.

    Returns:
        A tuple of extracted features and corresponding labels.
    """

    datagen = ImageDataGenerator(rescale=1./255)
    generator = datagen.flow_from_directory(
        directory,
        target_size=(150, 150),
        batch_size=batch_size,
        class_mode='binary'
    )

    features = np.zeros((sample_count, 4, 4, 512))
    labels = np.zeros((sample_count))

    i = 0
    for inputs_batch, labels_batch in generator:
        features_batch = conv_base.predict(inputs_batch)
        features[i * batch_size: (i + 1) * batch_size] = features_batch
        labels[i * batch_size: (i + 1) * batch_size] = labels_batch
        i += 1

        if i * batch_size >= sample_count:
            # Break after processing all samples
            break

    return features, labels

# Assuming you have defined `conv_base` and have the necessary image directories
train_features, train_labels = extract_features(train_dir, 2000)
validation_features, validation_labels = extract_features(validation_dir, 1000)
test_features, test_labels = extract_features(test_dir, 1000)