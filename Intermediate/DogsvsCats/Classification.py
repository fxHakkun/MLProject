import tensorflow as tf
import keras

test_dir = "D:\_BACKUP\Documents\Belajar\MachineLearning_AI_Project\Beginner\DogsvsCats\test"
train_dir = "D:\_BACKUP\Documents\Belajar\MachineLearning_AI_Project\Beginner\DogsvsCats\train"

train_ds = keras.utils.image_dataset_from_directory(
    directory=train_dir,
    labels='inferred',
    label_mode='categorical',
    batch_size=32,
    image_size=(256, 256))
test_ds = keras.utils.image_dataset_from_directory(
    directory=test_dir,
    labels='inferred',
    label_mode='categorical',
    batch_size=32,
    image_size=(256, 256))