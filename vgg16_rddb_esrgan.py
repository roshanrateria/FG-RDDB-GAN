import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
import tensorflow_hub as hub

# --- Configuration ---
OUTPUT_CHANNELS = 3
IMG_WIDTH = 256
IMG_HEIGHT = 256
NUM_RRDB_BLOCKS = 3 # Based on loop range `for _ in range(3)` in OCR
RRDB_FILTERS = 5 # OCR RRDB call uses filters=5 -> use 5 here, though seems low
VGG_FEATURE_FILTERS = 512 # Number of filters after downsampling VGG features

# Label embedding dims
NUM_EMOTION_CLASSES = 6
NUM_ORIENTATION_CLASSES = 3
LABEL_EMBED_DIM = 256

# Loss weights from OCR for this specific model (VGG+RDDB+ESRGAN)
GAN_LOSS_WEIGHT = 1 # Default if not specified, adjust as needed
L1_LOSS_WEIGHT = 100
DICE_LOSS_WEIGHT = 100
PERCEPTUAL_LOSS_WEIGHT = 100
EDGE_LOSS_WEIGHT = 5 # New edge loss component

ESRGAN_MODEL_URL = "https://kaggle.com/models/kaggle/esrgan-tf2/frameworks/TensorFlow2/variations/esrgan-tf2/versions/1"

# --- Building Blocks (Downsample, Upsample, RRDB) ---
# Using RRDB with scaling factor like in ESRGAN paper, as implied by 'ESRGAN with other layers' description
# However, the direct VGG+RRDB+ESRGAN block doesn't show scaling. Let's stick to Add() for RRDB here.

def downsample(filters, size, apply_batchnorm=True):
    initializer = tf.random_normal_initializer(0., 0.02)
    result = tf.keras.Sequential()
    result.add(layers.Conv2D(filters, size, strides=2, padding='same', kernel_initializer=initializer, use_bias=False))
    if apply_batchnorm: result.add(layers.BatchNormalization())
    result.add(layers.LeakyReLU())
    return result

def upsample(filters, size, apply_dropout=False):
    initializer = tf.random_normal_initializer(0., 0.02)
    result = tf.keras.Sequential()
    result.add(layers.Conv2DTranspose(filters, size, strides=2, padding='same', kernel_initializer=initializer, use_bias=False))
    result.add(layers.BatchNormalization())
    if apply_dropout: result.add(layers.Dropout(0.5))
    result.add(layers.ReLU())
    return result

# RRDB Block (Simple Add version based on direct OCR interpretation)
def residual_dense_block(x, filters, kernel_size=3):
    initializer = tf.random_normal_initializer(0., 0.02)
    input_features = x
    for _ in range(3): # 3 dense layers
        out = layers.Conv2D(filters, kernel_size=kernel_size, padding='same', kernel_initializer=initializer)(x)
        out = layers.LeakyReLU()(out)
        x = layers.Concatenate()([x, out])
    x = layers.Conv2D(filters, kernel_size=kernel_size, padding='same', kernel_initializer=initializer)(x) # Feature fusion
    return x

def RRDB(x, filters, kernel_size=3):
    input_features = x
    processed_features = residual_dense_block(x, filters, kernel_size)
    return layers.Add()([input_features, processed_features])


# --- VGG Feature Extractor ---
def build_vgg_extractor():
    vgg = VGG16(include_top=False, weights='imagenet', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3))
    vgg.trainable = False
    # Use a mid-level layer, e.g., block3_conv3 or block4_conv3
    # The OCR code implies downsampling it twice with stride 2 (512 filters), maybe from block5?
    # Let's assume block5_conv3 output (16x16x512 for 256 input) and downsample once more if needed.
    # Or maybe use earlier layers and downsample more.
    # Let's try block3_conv3 (64x64x256) and downsample it.
    layer_name = 'block3_conv3' # Example layer
    vgg_model = Model(inputs=vgg.input, outputs=vgg.get_layer(layer_name).output, name="vgg_extractor")
    return vgg_model

vgg_extractor = build_vgg_extractor()


# --- Generator with VGG Features + RRDBs + ESRGAN ---
def Generator(img_height=IMG_HEIGHT, img_width=IMG_WIDTH,
              num_emotion_classes=NUM_EMOTION_CLASSES,
              num_orientation_classes=NUM_ORIENTATION_CLASSES,
              num_rrdbs=NUM_RRDB_BLOCKS, rddb_filters=RRDB_FILTERS,
              vgg_filters=VGG_FEATURE_FILTERS):

    inputs = layers.Input(shape=[img_height, img_width, 3], name='input_image')

    # Label Embeddings
    inp_emotion_label = layers.Input(shape=[1,], name='emotion_label')
    label_embedding_emotion = layers.Embedding(num_emotion_classes, LABEL_EMBED_DIM)(inp_emotion_label)
    label_embedding_emotion = layers.Flatten()(label_embedding_emotion)
    label_embedding_emotion = layers.Dense(img_height * img_width * 1)(label_embedding_emotion)
    label_embedding_emotion = layers.Reshape((img_height, img_width, 1))(label_embedding_emotion)

    inp_orientation_label = layers.Input(shape=[1,], name='orientation_label')
    label_embedding_orientation = layers.Embedding(num_orientation_classes, LABEL_EMBED_DIM)(inp_orientation_label)
    label_embedding_orientation = layers.Flatten()(label_embedding_orientation)
    label_embedding_orientation = layers.Dense(img_height * img_width * 1)(label_embedding_orientation)
    label_embedding_orientation = layers.Reshape((img_height, img_width, 1))(label_embedding_orientation)

    # --- VGG Feature Path ---
    # Preprocess input for VGG
    input_vgg = tf.keras.applications.vgg16.preprocess_input(inputs * 255.0) # Assuming input is [0,1] or [-1,1] -> scale to VGG input
    vgg_features = vgg_extractor(input_vgg) # e.g., (batch, 64, 64, 256)
    # Downsample VGG features (OCR: `downsample(512,4)` applied twice)
    vgg_f_down1 = downsample(vgg_filters, 4)(vgg_features) # (batch, 32, 32, 512)
    vgg_f_down2 = downsample(vgg_filters, 4)(vgg_f_down1) # (batch, 16, 16, 512) - Target size?

    # --- Main Path (U-Net structure) ---
    x = layers.Concatenate()([inputs, label_embedding_emotion, label_embedding_orientation])

    down_stack = [ # Standard U-Net downsampling
        downsample(64, 4, apply_batchnorm=False), # (128, 128, 64)
        downsample(128, 4),                      # (64, 64, 128)
        downsample(256, 4),                      # (32, 32, 256)
        downsample(512, 4),                      # (16, 16, 512) # Match VGG feature size here
        # Continue downsampling for bottleneck
        downsample(512, 4),                      # (8, 8, 512)
        downsample(512, 4),                      # (4, 4, 512)
        downsample(512, 4),                      # (2, 2, 512)
        # downsample(512, 4),                      # (1, 1, 512) # Optional deeper bottleneck
    ]
    up_stack = [ # Standard U-Net upsampling
        # upsample(512, 4, apply_dropout=True),   # From 1x1
        upsample(512, 4, apply_dropout=True),   # From 2x2
        upsample(512, 4, apply_dropout=True),   # From 4x4
        upsample(512, 4),                       # From 8x8 -> 16x16
        # --- Inject VGG Features Here? ---
        # After upsampling to 16x16, before concat with skip connection?
        # Or concat VGG features with the skip connection itself?
        # OCR: `x = tf.keras.layers.Concatenate()([x, vgg_layer1_out])` applied after RRDBs.
        # Let's follow OCR logic: Inject VGG after RRDBs in bottleneck.

        upsample(256, 4),                       # 16x16 -> 32x32
        upsample(128, 4),                       # 32x32 -> 64x64
        upsample(64, 4),                        # 64x64 -> 128x128
    ]

    skips = []
    x_main = x
    for i, down in enumerate(down_stack):
        x_main = down(x_main)
        skips.append(x_main)
        # If VGG features injected at 16x16 level (after 4 downsamples)
        # if i == 3: # After downsample to 16x16x512
        #     x_main = layers.Concatenate()([x_main, vgg_f_down2]) # Inject here? No, OCR injects later.
        #     print("Injecting VGG features at 16x16")

    # --- Bottleneck with RRDBs ---
    x_bottleneck = x_main # Output from down_stack
    # OCR: `for _ in range(3): x = RRDB(x, 5)` - Apply RRDBs
    for _ in range(NUM_RRDB_BLOCKS):
         x_bottleneck = RRDB(x_bottleneck, filters=x_bottleneck.shape[-1]) # Use bottleneck filters

    # --- Inject VGG Features after RRDBs (as per OCR code line) ---
    # Dimensions must match. Bottleneck is e.g., 2x2x512. VGG features are 16x16x512. Mismatch!
    # Re-reading OCR: `vgg_layer1_out = vgg(inputs)` then downsampled twice.
    # `x = tf.keras.layers.Concatenate()([x, vgg_layer1_out])` happens *after* the RRDB loop.
    # This implies x (output of RRDB) and vgg_layer1_out must have compatible shapes.
    # This structure seems problematic in the OCR.

    # --- Let's assume a more logical injection point: Concat VGG features with skip connection ---
    x = x_bottleneck # Start decoding from bottleneck
    skips = reversed(skips[:-1]) # Skips from U-Net path

    for i, (up, skip) in enumerate(zip(up_stack, skips)):
        x = up(x)
        # Inject VGG features when skip connection size matches (e.g., 16x16)
        if x.shape[1] == vgg_f_down2.shape[1] and x.shape[2] == vgg_f_down2.shape[2]:
             print(f"Injecting VGG features at size {x.shape[1]}x{x.shape[2]}")
             skip_combined = layers.Concatenate()([skip, vgg_f_down2])
             x = layers.Concatenate()([x, skip_combined])
        else:
             x = layers.Concatenate()([x, skip])


    # Final Layer before ESRGAN
    initializer = tf.random_normal_initializer(0., 0.02)
    # OCR uses 'tanh' activation here. Let's assume this version reverted back.
    last = layers.Conv2DTranspose(OUTPUT_CHANNELS, 4, strides=2,
                                         padding='same', kernel_initializer=initializer,
                                         activation='tanh')(x) # Output: (batch, 256, 256, 3), range [-1, 1]

    # Post-processing before ESRGAN (as per OCR for 'ESRGAN with other layers')
    # x = (last + 1.0) * 127.5 # Scale [-1, 1] to [0, 255]

    # Apply ESRGAN Layer
    # ESRGAN layer expects specific input range, often uint8 [0, 255] or float [0, 1]
    # If input is tanh [-1, 1], need scaling
    last_scaled_01 = (last + 1.0) / 2.0 # Scale [-1, 1] to [0, 1]

    esrgan_layer = hub.KerasLayer(ESRGAN_MODEL_URL, trainable=False)
    output_esrgan = esrgan_layer(last_scaled_01) # Input should be float [0, 1] or similar

    # Clip and Scale ESRGAN output (typically [0, 255] float) back to [-1, 1] for loss consistency
    output_clipped = tf.clip_by_value(output_esrgan, 0, 255)
    output_final = (output_clipped / 127.5) - 1.0 # Scale [0, 255] to [-1, 1]

    return tf.keras.Model(inputs=[inputs, inp_emotion_label, inp_orientation_label], outputs=output_final)


# --- Discriminator ---
# OCR: "Same code as last attempt" -> Use rddb_esrgan's discriminator
def Discriminator(img_height=IMG_HEIGHT, img_width=IMG_WIDTH,
                  num_emotion_classes=NUM_EMOTION_CLASSES,
                  num_orientation_classes=NUM_ORIENTATION_CLASSES):
    # ... (Copy Discriminator code from rddb_esrgan.py / rddb_gan.py) ...
    initializer = tf.random_normal_initializer(0., 0.02)
    inp = layers.Input(shape=[img_height, img_width, 3], name='input_image')
    tar = layers.Input(shape=[img_height, img_width, 3], name='target_image')
    inp_emotion_label = layers.Input(shape=[1,], name='emotion_label')
    label_embedding_emotion = layers.Embedding(num_emotion_classes, LABEL_EMBED_DIM)(inp_emotion_label)
    label_embedding_emotion = layers.Flatten()(label_embedding_emotion)
    label_embedding_emotion = layers.Dense(img_height * img_width * 1)(label_embedding_emotion)
    label_embedding_emotion = layers.Reshape((img_height, img_width, 1))(label_embedding_emotion)
    inp_orientation_label = layers.Input(shape=[1,], name='orientation_label')
    label_embedding_orientation = layers.Embedding(num_orientation_classes, LABEL_EMBED_DIM)(inp_orientation_label)
    label_embedding_orientation = layers.Flatten()(label_embedding_orientation)
    label_embedding_orientation = layers.Dense(img_height * img_width * 1)(label_embedding_orientation)
    label_embedding_orientation = layers.Reshape((img_height, img_width, 1))(label_embedding_orientation)
    x = layers.concatenate([inp, tar, label_embedding_emotion, label_embedding_orientation])
    down1 = downsample(64, 4, False)(x); down2 = downsample(128, 4)(down1); down3 = downsample(256, 4)(down2)
    zero_pad1 = layers.ZeroPadding2D()(down3)
    conv = layers.Conv2D(512, 4, strides=1, kernel_initializer=initializer, use_bias=False)(zero_pad1)
    batchnorm1 = layers.BatchNormalization()(conv); leaky_relu = layers.LeakyReLU()(batchnorm1)
    zero_pad2 = layers.ZeroPadding2D()(leaky_relu)
    last = layers.Conv2D(1, 4, strides=1, kernel_initializer=initializer)(zero_pad2)
    return tf.keras.Model(inputs=[inp, tar, inp_emotion_label, inp_orientation_label], outputs=last)


# --- Loss Functions ---
loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)
# VGG model for perceptual loss
def build_vgg():
    vgg = VGG16(include_top=False, weights='imagenet', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3))
    vgg.trainable = False
    layer_name = 'block3_conv3'
    vgg_model = Model(inputs=vgg.input, outputs=vgg.get_layer(layer_name).output, name="vgg_perceptual")
    return vgg_model
vgg_perceptual_model = build_vgg()

def perceptual_loss(y_true, y_pred):
     # Assuming y_true, y_pred are [-1, 1]
    y_true_vgg = tf.keras.applications.vgg16.preprocess_input((y_true + 1.0) * 127.5)
    y_pred_vgg = tf.keras.applications.vgg16.preprocess_input((y_pred + 1.0) * 127.5)
    y_true_features = vgg_perceptual_model(y_true_vgg)
    y_pred_features = vgg_perceptual_model(y_pred_vgg)
    return tf.reduce_mean(tf.abs(y_true_features - y_pred_features))

# Dice Loss
def dice_loss(y_true, y_pred, smooth=1e-5):
    # Scale [-1, 1] to [0, 1] for Dice which assumes probabilities/segmentations
    y_true_01 = (y_true + 1.0) / 2.0
    y_pred_01 = (y_pred + 1.0) / 2.0
    intersection = tf.reduce_sum(y_true_01 * y_pred_01, axis=[1, 2, 3])
    union = tf.reduce_sum(y_true_01, axis=[1, 2, 3]) + tf.reduce_sum(y_pred_01, axis=[1, 2, 3])
    dice = (2. * intersection + smooth) / (union + smooth)
    return 1.0 - tf.reduce_mean(dice)

# Edge Loss (Sobel filter)
def edge_loss(y_true, y_pred):
    # Assuming y_true, y_pred are [-1, 1], convert to grayscale [0, 1]
    y_true_gray = tf.image.rgb_to_grayscale((y_true + 1.0) / 2.0)
    y_pred_gray = tf.image.rgb_to_grayscale((y_pred + 1.0) / 2.0)

    # Sobel edges
    gy_true, gx_true = tf.image.image_gradients(y_true_gray)
    gy_pred, gx_pred = tf.image.image_gradients(y_pred_gray)

    # L1 loss on gradients
    loss_gx = tf.reduce_mean(tf.abs(gx_true - gx_pred))
    loss_gy = tf.reduce_mean(tf.abs(gy_true - gy_pred))

    return loss_gx + loss_gy

def discriminator_loss(disc_real_output, disc_generated_output):
    real_loss = loss_object(tf.ones_like(disc_real_output), disc_real_output)
    generated_loss = loss_object(tf.zeros_like(disc_generated_output), disc_generated_output)
    total_disc_loss = real_loss + generated_loss
    return total_disc_loss

def generator_loss(disc_generated_output, gen_output, target):
    gan_loss = loss_object(tf.ones_like(disc_generated_output), disc_generated_output)
    l1_loss = tf.reduce_mean(tf.abs(target - gen_output))
    dice_loss_value = dice_loss(target, gen_output)
    perceptual_loss_value = perceptual_loss(target, gen_output)
    edge_loss_value = edge_loss(target, gen_output)

    total_gen_loss = (gan_loss + # GAN weight seems to be 1 here based on sum format
                      L1_LOSS_WEIGHT * l1_loss +
                      DICE_LOSS_WEIGHT * dice_loss_value +
                      PERCEPTUAL_LOSS_WEIGHT * perceptual_loss_value +
                      EDGE_LOSS_WEIGHT * edge_loss_value)

    return total_gen_loss, gan_loss, l1_loss # Return main components

# --- Example Usage ---
if __name__ == '__main__':
    generator = Generator()
    tf.keras.utils.plot_model(generator, to_file='vgg16_rddb_esrgan_generator.png', show_shapes=True, dpi=64)

    discriminator = Discriminator()
    tf.keras.utils.plot_model(discriminator, to_file='vgg16_rddb_esrgan_discriminator.png', show_shapes=True, dpi=64)

    print("VGG16 + RRDB + ESRGAN model definitions loaded.")
    print(f"Generator inputs: {generator.input_names}")
    print(f"Discriminator inputs: {discriminator.input_names}")

    # --- Placeholder for Training Loop ---
    # Ensure target images are in [-1, 1] range for losses
