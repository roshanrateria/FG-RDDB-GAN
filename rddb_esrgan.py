import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.models import Model
import tensorflow_hub as hub

# --- Configuration ---
OUTPUT_CHANNELS = 3
IMG_WIDTH = 256
IMG_HEIGHT = 256
NUM_RRDB_BLOCKS = 5 # OCR: Add 5 RRDBs (used 3 in range loop, inconsistent) -> using 3 here based on loop
RRDB_FILTERS = 64 # Filter count matching first downsample layer

# Label embedding dims
NUM_EMOTION_CLASSES = 6
NUM_ORIENTATION_CLASSES = 3
LABEL_EMBED_DIM = 256

# Loss weights from OCR for this specific model (RDDB+ESRGAN)
GAN_LOSS_WEIGHT = 5
L1_LOSS_WEIGHT = 100
DICE_LOSS_WEIGHT = 150 # Changed from 70 in RRDB baseline
PERCEPTUAL_LOSS_WEIGHT = 120

ESRGAN_MODEL_URL = "https://kaggle.com/models/kaggle/esrgan-tf2/frameworks/TensorFlow2/variations/esrgan-tf2/versions/1" # Kaggle URL from OCR

# --- Building Blocks (Downsample, Upsample, RRDB - adapted from rddb_gan.py) ---

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
    # Simple residual connection (no scaling factor shown in this OCR block)
    return layers.Add()([input_features, processed_features])

# --- Generator with RRDBs + ESRGAN layer ---
def Generator(img_height=IMG_HEIGHT, img_width=IMG_WIDTH,
              num_emotion_classes=NUM_EMOTION_CLASSES,
              num_orientation_classes=NUM_ORIENTATION_CLASSES,
              num_rrdbs=NUM_RRDB_BLOCKS, rddb_filters=RRDB_FILTERS): # Use config defaults

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

    x = layers.Concatenate()([inputs, label_embedding_emotion, label_embedding_orientation])

    # --- U-Net structure based on OCR code block ---
    # Downsampling Path (copied from OCR)
    down_stack = [
        downsample(64, 4, apply_batchnorm=False), # (128, 128, 64)
        downsample(128, 4),                      # (64, 64, 128)
        downsample(256, 4),                      # (32, 32, 256)
        downsample(512, 4),                      # (16, 16, 512)
        downsample(512, 4),                      # (8, 8, 512)
        downsample(512, 4),                      # (4, 4, 512)
        downsample(512, 4),                      # (2, 2, 512)
        downsample(512, 4),                      # (1, 1, 512) # Bottleneck
    ]
    # Upsampling Path (copied from OCR, but misses last few layers for ESRGAN variant?)
    up_stack = [
        upsample(512, 4, apply_dropout=True),    # (2, 2, 1024)
        upsample(512, 4, apply_dropout=True),    # (4, 4, 1024)
        upsample(512, 4, apply_dropout=True),    # (8, 8, 1024)
        upsample(512, 4),                         # (16, 16, 1024)
        upsample(256, 4),                         # (32, 32, 512)
        # Missing upsample(128, 4) and upsample(64, 4) compared to standard U-Net?
        # Let's assume full U-Net structure before ESRGAN layer
        # Adding the missing layers based on typical U-Net
        upsample(128, 4),                         # (64, 64, 256)
        upsample(64, 4),                          # (128, 128, 128)
    ]

    # Add RRDB Blocks (OCR: `for _ in range(3): x = RRDB(x, 5)`)
    # Where to add them? Let's assume bottleneck like in rddb_gan.py's inferred structure
    # This contradicts the literal OCR code for this section too, which applies them first.
    # --- Sticking to the literal (if strange) OCR structure for *this* file: ---
    x_rrdbs = layers.Conv2D(rddb_filters, 3, padding='same')(x) # Initial conv
    # OCR uses `range(3)` for RRDB loop here
    for _ in range(3):
         x_rrdbs = RRDB(x_rrdbs, filters=rddb_filters)

    # The OCR code doesn't seem to *use* x_rrdbs in the main path. It restarts 'x'.
    # Let's ignore the RRDB definition here and follow the U-Net + ESRGAN structure shown.

    skips = []
    x_unet = x # Start U-Net path from concatenated input
    for down in down_stack:
        x_unet = down(x_unet)
        skips.append(x_unet)

    skips = reversed(skips[:-1]) # Exclude bottleneck

    # Upsampling and establishing skip connections
    for up, skip in zip(up_stack, skips):
        x_unet = up(x_unet)
        x_unet = layers.Concatenate()([x_unet, skip])

    # Final Layer before ESRGAN
    initializer = tf.random_normal_initializer(0., 0.02)
    # Output needs to match ESRGAN input requirements if any, otherwise standard ConvTranspose
    # The OCR uses 'sigmoid' activation here.
    last = layers.Conv2DTranspose(OUTPUT_CHANNELS, 4, strides=2,
                                         padding='same', kernel_initializer=initializer,
                                         activation='sigmoid')(x_unet) # Output: (batch, 256, 256, 3), range [0, 1]

    # Apply ESRGAN Layer
    # Input to ESRGAN is typically uint8 [0, 255] or float [0, 1]
    # Since activation is sigmoid ([0, 1]), it might be directly compatible.
    esrgan_layer = hub.KerasLayer(ESRGAN_MODEL_URL, trainable=False) # Load pre-trained ESRGAN
    output = esrgan_layer(last) # Apply ESRGAN for super-resolution/enhancement

    # ESRGAN output is typically float32 in [0, 255]. Need to scale back to [-1, 1] for tanh compatibility if needed by losses.
    # Or adjust losses to expect [0, 1] or [0, 255].
    # Let's assume loss functions handle the range [0, 255] or scale output back.
    # For consistency with previous models using 'tanh', let's scale to [-1, 1]
    # output_scaled = (output / 127.5) - 1.0
    # However, the next model description implies the output is clipped 0-255 and divided by 255.
    # Let's follow that for consistency forward:
    output_clipped = tf.clip_by_value(output, 0, 255)
    output_final = output_clipped / 255.0 # Scale to [0, 1]

    # If losses expect [-1, 1] (like perceptual often does with VGG default)
    # output_final = (output_clipped / 127.5) - 1.0

    return tf.keras.Model(inputs=[inputs, inp_emotion_label, inp_orientation_label], outputs=output_final)


# --- Discriminator (PatchGAN with Label Embedding) ---
# Assume same discriminator as rddb_gan.py
def Discriminator(img_height=IMG_HEIGHT, img_width=IMG_WIDTH,
                  num_emotion_classes=NUM_EMOTION_CLASSES,
                  num_orientation_classes=NUM_ORIENTATION_CLASSES):
    # ... (Copy Discriminator code from rddb_gan.py) ...
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
     # Ensure inputs are scaled appropriately for VGG, e.g., [0, 1] -> VGG range
     # Assuming y_true/y_pred are [0, 1] from sigmoid/ESRGAN output scaling
    y_true_vgg = tf.keras.applications.vgg16.preprocess_input(y_true * 255.0)
    y_pred_vgg = tf.keras.applications.vgg16.preprocess_input(y_pred * 255.0)
    y_true_features = vgg_perceptual_model(y_true_vgg)
    y_pred_features = vgg_perceptual_model(y_pred_vgg)
    return tf.reduce_mean(tf.abs(y_true_features - y_pred_features))

# Dice Loss
def dice_loss(y_true, y_pred, smooth=1e-5):
    intersection = tf.reduce_sum(y_true * y_pred, axis=[1, 2, 3])
    union = tf.reduce_sum(y_true, axis=[1, 2, 3]) + tf.reduce_sum(y_pred, axis=[1, 2, 3])
    dice = (2. * intersection + smooth) / (union + smooth)
    return 1.0 - tf.reduce_mean(dice)

def discriminator_loss(disc_real_output, disc_generated_output):
    real_loss = loss_object(tf.ones_like(disc_real_output), disc_real_output)
    generated_loss = loss_object(tf.zeros_like(disc_generated_output), disc_generated_output)
    total_disc_loss = real_loss + generated_loss
    return total_disc_loss

def generator_loss(disc_generated_output, gen_output, target):
    # Ensure target is scaled same as gen_output, e.g. [0, 1]
    target_scaled = (target + 1.0) / 2.0 # Assuming target was [-1, 1]
    target_scaled = tf.clip_by_value(target_scaled, 0.0, 1.0) # Clip just in case

    gan_loss = loss_object(tf.ones_like(disc_generated_output), disc_generated_output)
    l1_loss = tf.reduce_mean(tf.abs(target_scaled - gen_output)) # Use scaled target
    dice_loss_value = dice_loss(target_scaled, gen_output) # Use scaled target
    perceptual_loss_value = perceptual_loss(target_scaled, gen_output) # Use scaled target

    total_gen_loss = (GAN_LOSS_WEIGHT * gan_loss +
                      L1_LOSS_WEIGHT * l1_loss +
                      DICE_LOSS_WEIGHT * dice_loss_value +
                      PERCEPTUAL_LOSS_WEIGHT * perceptual_loss_value)

    return total_gen_loss, gan_loss, l1_loss # Return main components

# --- Example Usage ---
if __name__ == '__main__':
    generator = Generator()
    tf.keras.utils.plot_model(generator, to_file='rddb_esrgan_generator.png', show_shapes=True, dpi=64)

    discriminator = Discriminator()
    tf.keras.utils.plot_model(discriminator, to_file='rddb_esrgan_discriminator.png', show_shapes=True, dpi=64)

    print("RRDB + ESRGAN model definitions loaded.")
    print(f"Generator inputs: {generator.input_names}")
    print(f"Discriminator inputs: {discriminator.input_names}")

    # --- Placeholder for Training Loop ---
    # Ensure target images provided to generator_loss are scaled to [0, 1]
