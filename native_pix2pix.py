import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.models import Model

# --- Configuration ---
OUTPUT_CHANNELS = 3
LAMBDA = 100 # Weight for L1 loss (in original paper, but weights adjusted below)
IMG_WIDTH = 256
IMG_HEIGHT = 256

# Loss weights from OCR for this specific model
GAN_LOSS_WEIGHT = 50
L1_LOSS_WEIGHT = 100
DICE_LOSS_WEIGHT = 150
PERCEPTUAL_LOSS_WEIGHT = 200

# --- Building Blocks (Standard Pix2pix) ---

def downsample(filters, size, apply_batchnorm=True):
    initializer = tf.random_normal_initializer(0., 0.02)
    result = tf.keras.Sequential()
    result.add(
        layers.Conv2D(filters, size, strides=2, padding='same',
                             kernel_initializer=initializer, use_bias=False))
    if apply_batchnorm:
        result.add(layers.BatchNormalization())
    result.add(layers.LeakyReLU())
    return result

def upsample(filters, size, apply_dropout=False):
    initializer = tf.random_normal_initializer(0., 0.02)
    result = tf.keras.Sequential()
    result.add(
        layers.Conv2DTranspose(filters, size, strides=2,
                                        padding='same',
                                        kernel_initializer=initializer,
                                        use_bias=False))
    result.add(layers.BatchNormalization())
    if apply_dropout:
        result.add(layers.Dropout(0.5))
    result.add(layers.ReLU())
    return result

# --- Generator (Standard U-Net) ---

def Generator(img_height=IMG_HEIGHT, img_width=IMG_WIDTH, output_channels=OUTPUT_CHANNELS):
    inputs = layers.Input(shape=[img_height, img_width, 3])

    down_stack = [
        downsample(64, 4, apply_batchnorm=False), # (batch_size, 128, 128, 64)
        downsample(128, 4),                      # (batch_size, 64, 64, 128)
        downsample(256, 4),                      # (batch_size, 32, 32, 256)
        downsample(512, 4),                      # (batch_size, 16, 16, 512)
        downsample(512, 4),                      # (batch_size, 8, 8, 512)
        downsample(512, 4),                      # (batch_size, 4, 4, 512)
        downsample(512, 4),                      # (batch_size, 2, 2, 512)
        downsample(512, 4),                      # (batch_size, 1, 1, 512)
    ]

    up_stack = [
        upsample(512, 4, apply_dropout=True),    # (batch_size, 2, 2, 1024)
        upsample(512, 4, apply_dropout=True),    # (batch_size, 4, 4, 1024)
        upsample(512, 4, apply_dropout=True),    # (batch_size, 8, 8, 1024)
        upsample(512, 4),                         # (batch_size, 16, 16, 1024)
        upsample(256, 4),                         # (batch_size, 32, 32, 512)
        upsample(128, 4),                         # (batch_size, 64, 64, 256)
        upsample(64, 4),                          # (batch_size, 128, 128, 128)
    ]

    initializer = tf.random_normal_initializer(0., 0.02)
    last = layers.Conv2DTranspose(output_channels, 4,
                                         strides=2,
                                         padding='same',
                                         kernel_initializer=initializer,
                                         activation='tanh') # (batch_size, 256, 256, 3)

    x = inputs

    # Downsampling through the model
    skips = []
    for down in down_stack:
        x = down(x)
        skips.append(x)

    skips = reversed(skips[:-1])

    # Upsampling and establishing the skip connections
    for up, skip in zip(up_stack, skips):
        x = up(x)
        x = layers.Concatenate()([x, skip])

    x = last(x)

    return tf.keras.Model(inputs=inputs, outputs=x)

# --- Discriminator (Standard PatchGAN) ---

def Discriminator(img_height=IMG_HEIGHT, img_width=IMG_WIDTH):
    initializer = tf.random_normal_initializer(0., 0.02)

    inp = layers.Input(shape=[img_height, img_width, 3], name='input_image')
    tar = layers.Input(shape=[img_height, img_width, 3], name='target_image')

    x = layers.concatenate([inp, tar]) # (batch_size, 256, 256, channels*2)

    down1 = downsample(64, 4, False)(x) # (batch_size, 128, 128, 64)
    down2 = downsample(128, 4)(down1)    # (batch_size, 64, 64, 128)
    down3 = downsample(256, 4)(down2)    # (batch_size, 32, 32, 256)

    zero_pad1 = layers.ZeroPadding2D()(down3) # (batch_size, 34, 34, 256)
    conv = layers.Conv2D(512, 4, strides=1,
                                kernel_initializer=initializer,
                                use_bias=False)(zero_pad1) # (batch_size, 31, 31, 512)

    batchnorm1 = layers.BatchNormalization()(conv)
    leaky_relu = layers.LeakyReLU()(batchnorm1)
    zero_pad2 = layers.ZeroPadding2D()(leaky_relu) # (batch_size, 33, 33, 512)

    last = layers.Conv2D(1, 4, strides=1,
                                kernel_initializer=initializer)(zero_pad2) # (batch_size, 30, 30, 1)

    return tf.keras.Model(inputs=[inp, tar], outputs=last)

# --- Loss Functions ---
loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def discriminator_loss(disc_real_output, disc_generated_output):
    real_loss = loss_object(tf.ones_like(disc_real_output), disc_real_output)
    generated_loss = loss_object(tf.zeros_like(disc_generated_output), disc_generated_output)
    total_disc_loss = real_loss + generated_loss
    return total_disc_loss

# --- Additional Loss Components ---

# Perceptual Loss (VGG based)
def build_vgg():
    vgg = VGG16(include_top=False, weights='imagenet', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3))
    vgg.trainable = False
    layer_name = 'block3_conv3' # Choose a layer for feature extraction
    vgg_model = Model(inputs=vgg.input, outputs=vgg.get_layer(layer_name).output, name="vgg_perceptual")
    return vgg_model

vgg_model = build_vgg() # Instantiate once

def perceptual_loss(y_true, y_pred):
    # Ensure inputs are float32 and scaled [0, 1] if needed, or VGG preprocess_input
    # Assuming inputs are already in suitable range for VGG (e.g., scaled [-1, 1] -> [0, 255])
    # Or apply tf.keras.applications.vgg16.preprocess_input if needed
    y_true_features = vgg_model(y_true)
    y_pred_features = vgg_model(y_pred)
    return tf.reduce_mean(tf.abs(y_true_features - y_pred_features))

# Dice Loss
def dice_loss(y_true, y_pred, smooth=1e-5):
    """
    Dice loss function. Assumes inputs are float tensors (e.g., sigmoid output).
    Args:
        y_true: Ground truth segmentation mask. Shape (batch, H, W, C)
        y_pred: Predicted segmentation mask. Shape (batch, H, W, C)
        smooth: Smoothing factor to avoid division by zero.
    Returns:
        Dice loss value (scalar).
    """
    # Flatten spatial dimensions, keep batch and channel
    # y_true_f = tf.reshape(y_true, [tf.shape(y_true)[0], -1, tf.shape(y_true)[-1]])
    # y_pred_f = tf.reshape(y_pred, [tf.shape(y_pred)[0], -1, tf.shape(y_pred)[-1]])
    # For image generation, axis reduction is usually across H, W, C
    intersection = tf.reduce_sum(y_true * y_pred, axis=[1, 2, 3])
    union = tf.reduce_sum(y_true, axis=[1, 2, 3]) + tf.reduce_sum(y_pred, axis=[1, 2, 3])
    dice = (2. * intersection + smooth) / (union + smooth)
    return 1.0 - tf.reduce_mean(dice) # Return 1 - Dice Similarity Coefficient


def generator_loss(disc_generated_output, gen_output, target):
    gan_loss = loss_object(tf.ones_like(disc_generated_output), disc_generated_output)
    # Mean absolute error (L1 Loss)
    l1_loss = tf.reduce_mean(tf.abs(target - gen_output))
    # Dice Loss
    dice_loss_value = dice_loss(target, gen_output)
    # Perceptual Loss
    perceptual_loss_value = perceptual_loss(target, gen_output)

    # Combine losses with weights from OCR
    total_gen_loss = (GAN_LOSS_WEIGHT * gan_loss +
                      L1_LOSS_WEIGHT * l1_loss +
                      DICE_LOSS_WEIGHT * dice_loss_value +
                      PERCEPTUAL_LOSS_WEIGHT * perceptual_loss_value)

    # Return individual components for monitoring if needed
    return total_gen_loss, gan_loss, l1_loss, dice_loss_value, perceptual_loss_value

# --- Example Usage (Illustrative) ---
if __name__ == '__main__':
    generator = Generator()
    # generator.summary()
    tf.keras.utils.plot_model(generator, to_file='native_pix2pix_generator.png', show_shapes=True, dpi=64)

    discriminator = Discriminator()
    # discriminator.summary()
    tf.keras.utils.plot_model(discriminator, to_file='native_pix2pix_discriminator.png', show_shapes=True, dpi=64)

    print("Native Pix2pix model definitions loaded.")

    # --- Placeholder for Training Loop ---
    # Similar to modified_pix2pix.py, but inputs are only images
    # Ensure target images are appropriately scaled for loss calculations (e.g., [-1, 1])
