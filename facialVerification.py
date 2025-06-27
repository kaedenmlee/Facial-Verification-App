# import standard dependencies
from tensorflow.keras.metrics import Precision, Recall
import cv2
import os
import random
import numpy as np
from matplotlib import pyplot as plt
import uuid

# import TensorFlow (use only tensorflow.keras)
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer, Conv2D, Dense, MaxPooling2D, Input, Flatten

# Avoid OOM errors by setting memory growth
gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


# Setup paths
POS_PATH = os.path.join('data', 'positive')
NEG_PATH = os.path.join('data', 'negative')
ANC_PATH = os.path.join('data', 'anchor')

# ---------------------------------------------------------------------------------------------------------------
# function to create more augmented images


def data_aug(img):
    # Convert numpy array to tensor if needed
    if isinstance(img, np.ndarray):
        img = tf.convert_to_tensor(img, dtype=tf.float32)

    # Normalize to 0-1 range if needed
    if tf.reduce_max(img) > 1.0:
        img = img / 255.0

    data = []
    for i in range(9):
        # Create unique seeds for each augmentation
        seed1 = [i + 1, i + 2]
        seed2 = [i + 3, i + 4]
        seed3 = [i + 5, i + 6]
        seed4 = [i + 7, i + 8]

        # Apply augmentations (all work with float32 in 0-1 range)
        augmented_img = tf.image.stateless_random_brightness(
            img, max_delta=0.02, seed=seed1)
        augmented_img = tf.image.stateless_random_contrast(
            augmented_img, lower=0.6, upper=1, seed=seed2)
        augmented_img = tf.image.stateless_random_flip_left_right(
            augmented_img, seed=seed3)
        augmented_img = tf.image.stateless_random_saturation(
            augmented_img, lower=0.9, upper=1, seed=seed4)

        # Convert back to uint8 for saving (0-255 range)
        augmented_img = tf.cast(augmented_img * 255.0, tf.uint8)

        data.append(augmented_img)

    return data

# Uncomment to create augmented images
# Process anchor images
# for file_name in os.listdir(ANC_PATH):
#     if file_name.lower().endswith(('.jpg', '.jpeg', '.png')):  # Only process image files
#         img_path = os.path.join(ANC_PATH, file_name)
#         img = cv2.imread(img_path)

#         if img is not None:  # Check if image was loaded successfully
#             augmented_imgs = data_aug(img)

#             for i, image in enumerate(augmented_imgs):
#                 output_path = os.path.join(ANC_PATH, f'{uuid.uuid1()}.jpg')
#                 cv2.imwrite(output_path, image.numpy())

# Process positive images
# for file_name in os.listdir(POS_PATH):
#     if file_name.lower().endswith(('.jpg', '.jpeg', '.png')):  # Only process image files
#         img_path = os.path.join(POS_PATH, file_name)
#         img = cv2.imread(img_path)

#         if img is not None:  # Check if image was loaded successfully
#             augmented_imgs = data_aug(img)

#             for i, image in enumerate(augmented_imgs):
#                 output_path = os.path.join(POS_PATH, f'{uuid.uuid1()}.jpg')
#                 cv2.imwrite(output_path, image.numpy())

# print("Data augmentation completed!")


# get image directories - change number of images for testing
anchor = tf.data.Dataset.list_files(
    os.path.join(ANC_PATH, '*.jpg')).take(4000)
positive = tf.data.Dataset.list_files(
    os.path.join(POS_PATH, '*.jpg')).take(4000)
negative = tf.data.Dataset.list_files(
    os.path.join(NEG_PATH, '*.jpg')).take(5000)

# ---------------------------------------------------------------------------------------------------------------


def preprocess(file_path):
    # read in image from file path
    byte_img = tf.io.read_file(file_path)
    # load in the image
    img = tf.io.decode_jpeg(byte_img)

    # preprocess the image
    img = tf.image.resize(img, (100, 100))
    img = img / 255.0
    return img


# build dataset
positives = tf.data.Dataset.zip(
    (anchor, positive, tf.data.Dataset.from_tensor_slices(tf.ones(len(anchor)))))
negatives = tf.data.Dataset.zip(
    (anchor, negative, tf.data.Dataset.from_tensor_slices(tf.zeros(len(anchor)))))
data = positives.concatenate(negatives)

# preprocess images


def preprocess_twin(input_img, validation_img, label):
    return (preprocess(input_img), preprocess(validation_img), label)


# build dataloader pipeline
data = data.map(preprocess_twin)
data = data.cache()
data = data.shuffle(buffer_size=10000)

# Training Partition
train_data = data.take(round(len(data)*.7))
train_data = train_data.batch(16)
train_data = train_data.prefetch(8)

# Testing partition
test_data = data.skip(round(len(data)*.7))
test_data = test_data.take(round(len(data)*.3))
test_data = test_data.batch(16)
test_data = test_data.prefetch(8)

# ---------------------------------------------------------------------------------------------------------------
# Build embedding layer


def make_embedding():
    inp = Input(shape=(100, 100, 3), name='input_image')

    # first block
    c1 = Conv2D(64, (10, 10), activation='relu')(inp)
    m1 = MaxPooling2D(64, (2, 2), padding='same')(c1)

    # second block
    c2 = Conv2D(128, (7, 7), activation='relu')(m1)
    m2 = MaxPooling2D(64, (2, 2), padding='same')(c2)

    # third block
    c3 = Conv2D(128, (4, 4), activation='relu')(m2)
    m3 = MaxPooling2D(64, (2, 2), padding='same')(c3)

    # fourth block
    c4 = Conv2D(256, (4, 4), activation='relu')(m3)
    f1 = Flatten()(c4)
    d1 = Dense(4096, activation='relu')(f1)

    return Model(inputs=inp, outputs=d1, name='embedding')


embedding = make_embedding()
# embedding.summary()

# Build distance layer


class L1Dist(Layer):
    # init method - inheritance
    def __init__(self, **kwargs):
        super().__init__()

    # similarity calc
    def call(self, inputs):
        input_embedding, validation_embedding = inputs
        return tf.math.abs(input_embedding - validation_embedding)

# ---------------------------------------------------------------------------------------------------------------
# make siamese model


def make_siamese_model():
    # Anchor image input in the network
    input_image = Input(name='input_img', shape=(100, 100, 3))

    # validation iamge in the network
    validation_image = Input(name='validation_img', shape=(100, 100, 3))

    # combine siamese distance components
    siamese_layer = L1Dist()
    siamese_layer._name = 'distance'
    distances = siamese_layer(
        [embedding(input_image), embedding(validation_image)])

    # classification layer
    classifier = Dense(1, activation='sigmoid')(distances)

    return Model(inputs=[input_image, validation_image], outputs=classifier, name='Siamese_Network')


siamese_model = make_siamese_model()
# siamese_model.summary()


# setup loss and optimizer
binary_cross_loss = tf.keras.losses.BinaryCrossentropy()
opt = tf.keras.optimizers.Adam(1e-4)  # learning rate

# checkpoint callbacks
checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(opt=opt, siamese_model=siamese_model)


# ---------------------------------------------------------------------------------------------------------------
# build train step
@tf.function
def train_step(batch):

    with tf.GradientTape() as tape:
        # get anchor and positive/negative image
        X = batch[:2]
        # get label
        y = batch[2]
        # forward pass
        yhat = siamese_model(X, training=True)
        # calculate loss
        loss = binary_cross_loss(y, yhat)
    print(loss)

    # calculate gradients
    grad = tape.gradient(loss, siamese_model.trainable_variables)
    # calculate updated weights and apply to siamese model
    opt.apply_gradients(zip(grad, siamese_model.trainable_variables))

    return loss


# build training loop
def train(data, EPOCHS):
    # loop through epochs
    for epoch in range(1, EPOCHS+1):
        print('\nEpoch {}/{}'.format(epoch, EPOCHS))
        progbar = tf.keras.utils.Progbar(len(data))

        # Create metrics
        r = Recall()
        p = Precision()

        # loop through batches
        for idx, batch in enumerate(data):
            # run train step
            loss = train_step(batch)
            yhat = siamese_model.predict(batch[:2])
            r.update_state(batch[2], yhat)
            p.update_state(batch[2], yhat)
            progbar.update(idx+1)
        print(loss.numpy(), r.result().numpy(), p.result().numpy())

        # save checkpoints
        if epoch % 10 == 0:
            checkpoint.save(file_prefix=checkpoint_prefix)


# train model, set number of epochs
EPOCHS = 50

# uncomment to train model
# train(train_data, EPOCHS)

# ---------------------------------------------------------------------------------------------------------------
# Evaluate Model - using precision, and recall from keras.metrics
test_input, test_val, y_true = test_data.as_numpy_iterator().next()
yhat = siamese_model.predict([test_input, test_val])

# postprocessing the results
[1 if prediction > 0.5 else 0 for prediction in yhat]

# Create a metric object
m = Recall()

# Calculate recall value
m.update_state(y_true, yhat)
# print("Recall: ", m.result().numpy())

# Calculate precision value
m = Precision()
m.update_state(y_true, yhat)
# print("Precision: ", m.result().numpy())

# visualize results
plt.subplot(1, 2, 1)
plt.imshow(test_input[2])
plt.subplot(1, 2, 2)
plt.imshow(test_val[2])
plt.show()

# save weights
siamese_model.save('siamesemodelv2.keras')
print("Saved model to disk")
print(siamese_model.summary())
print(siamese_model.get_weights())

# ---------------------------------------------------------------------------------------------------------------
# reload model
model = tf.keras.models.load_model('siamesemodelv2.keras', custom_objects={
    'L1Dist': L1Dist, 'BinaryCrossentropy': tf.keras.losses.BinaryCrossentropy})

# make predictions with reloaded model
# model.predict([test_input, test_val])
# model.summary()
# ---------------------------------------------------------------------------------------------------------------

# Verification Function


def verify(model, detection_threshold, verification_threshold):
    results = []
    # Define valid image extensions
    image_extensions = ('.jpg', '.jpeg', '.png', '.gif',
                        '.bmp', '.JPG', '.JPEG', '.PNG', '.GIF', '.BMP')
    verification_dir = os.path.join('application_data', 'verification_images')

    for image in os.listdir(verification_dir):
        # Skip hidden files and non-image files
        if image.startswith('.') or not image.endswith(image_extensions):
            continue
        input_img = preprocess(os.path.join(
            'application_data', 'input_image', 'input_image.jpg'))
        validation_img = preprocess(os.path.join(verification_dir, image))
        # Make predictions
        result = model.predict(
            list(np.expand_dims([input_img, validation_img], axis=1)))
        results.append(result)

    # Detection Threshold : metric above which a prediction is considered positive
    detection = np.sum(np.array(results) > detection_threshold)

    # Verification Threshold: proportion of positive predictions/ total positive samples
    verification = detection / len(results) if len(results) > 0 else 0
    verified = verification > verification_threshold

    return results, verified


# ---------------------------------------------------------------------------------------------------------------
# Real time verification
# Establish webcam - 0 for mac webcam
cap = cv2.VideoCapture(0)
crop_width, crop_height = 250, 250
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)  # Width
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)  # Height
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Get frame dimensions
    height, width, _ = frame.shape

    # Calculate center crop coordinates
    center_x, center_y = width // 2, height // 2
    x1 = center_x - crop_width // 2
    x2 = center_x + crop_width // 2
    y1 = center_y - crop_height // 2
    y2 = center_y + crop_height // 2

    # Ensure crop stays within frame bounds
    x1 = max(0, x1)
    x2 = min(width, x2)
    y1 = max(0, y1)
    y2 = min(height, y2)

    frame = frame[y1:y2, x1:x2]

    cv2.imshow('Verification', frame)

    # verify face
    if cv2.waitKey(1) & 0xFF == ord('v'):
        # save image into input folder
        cv2.imwrite(os.path.join('application_data',
                    'input_image', 'input_image.jpg'), frame)
        # run verification
        results, verified = verify(model, 0.8, 0.7)  # change thresholds here
        print(results, verified)
    # press q to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# release webcam and close windows
cap.release()
cv2.destroyAllWindows()
