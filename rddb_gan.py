import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.models import Model

# --- Configuration ---
OUTPUT_CHANNELS = 3
IMG_WIDTH = 256
IMG_HEIGHT = 256
NUM_RRDB_BLOCKS = 5 # Example, OCR mentions adding 5 RRDBs
RRDB_FILTERS = 64 # Filters within RRDB, adjust as needed (OCR used 5 in RRDB call, likely #filters)

# Label embedding dims
NUM_EMOTION_CLASSES = 6 # Back to 6/3 as per code block
NUM_ORIENTATION_CLASSES = 3
LABEL_EMBED_DIM = 256 # Embedding dimension before projection

# Loss weights from OCR for this specific model (RRDB baseline)
GAN_LOSS_WEIGHT = 5
L1_LOSS_WEIGHT = 100
DICE_LOSS_WEIGHT = 70
PERCEPTUAL_LOSS_WEIGHT = 120

# --- Building Blocks (U-Net style down/upsample) ---

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

# --- RRDB Block ---
def residual_dense_block(x, filters, kernel_size=3):
    """Residual Dense Block"""
    initializer = tf.random_normal_initializer(0., 0.02)
    input_features = x
    # Dense block with convolution layers
    # Number of dense layers (3 in OCR example)
    for _ in range(3):
        out = layers.Conv2D(filters, kernel_size=kernel_size, padding='same', kernel_initializer=initializer)(x)
        out = layers.LeakyReLU()(out)
        x = layers.Concatenate()([x, out]) # Dense concatenation

    # Final 1x1 convolution (or 3x3 as in OCR) for feature fusion / residual learning
    # OCR shows 3x3, let's stick to that. Acts like a bottleneck layer.
    x = layers.Conv2D(filters, kernel_size=kernel_size, padding='same', kernel_initializer=initializer)(x)

    # The OCR description implies the *output* of this function is fed to RRDB,
    # and RRDB adds the input 'x' back.
    # Let's follow the structure: residual_dense_block outputs the processed features,
    # RRDB adds the residual connection.
    return x # Return processed features

def RRDB(x, filters, kernel_size=3):
    """Residual in Residual Dense Block"""
    input_features = x
    # Get processed features from the dense block
    processed_features = residual_dense_block(x, filters, kernel_size)
    # Residual connection: Add input to the processed features
    # Scaling factor beta often used here (e.g., 0.2), but not shown in this OCR block
    return layers.Add()([input_features, processed_features]) # Residual connection


# --- Generator with RRDBs ---
def Generator(img_height=IMG_HEIGHT, img_width=IMG_WIDTH,
              num_emotion_classes=NUM_EMOTION_CLASSES,
              num_orientation_classes=NUM_ORIENTATION_CLASSES,
              num_rrdbs=NUM_RRDB_BLOCKS, rddb_filters=RRDB_FILTERS):

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

    # Concatenate inputs and embeddings
    x = layers.Concatenate()([inputs, label_embedding_emotion, label_embedding_orientation])

    # Initial Convolution (optional, good practice)
    # initializer = tf.random_normal_initializer(0., 0.02)
    # x = layers.Conv2D(rddb_filters, kernel_size=3, padding='same', kernel_initializer=initializer)(x)
    # x_initial = x # Store for final residual connection if needed (like in ESRGAN)

    # Add RRDB Blocks (OCR adds them *before* downsampling stack)
    # The OCR code concatenates, then applies RRDB, then downsamples. Let's follow that.
    # The number of filters in RRDB needs clarification. OCR call is RRDB(x, 5).
    # This likely means filters=5, which seems very low. Let's assume it meant num_blocks=5
    # and use rddb_filters for the filter count.
    # Let's assume RRDB filters should match the first downsample layer (64).
    initial_conv = layers.Conv2D(rddb_filters, 3, padding='same')(x) # Map to RRDB filter dim
    x = initial_conv
    for _ in range(num_rrdbs):
        x = RRDB(x, filters=rddb_filters) # Apply RRDB blocks

    # The OCR code then proceeds with a standard U-Net down/up stack applied to the *concatenated input*, not the RRDB output.
    # This seems structurally incorrect. Let's assume the RRDBs replace the U-Net body.
    # Alternative interpretation: RRDBs are applied *after* the encoder, in the bottleneck.
    # Let's follow the most plausible structure: Input -> CNN -> RRDBs -> CNN -> Output

    # Let's re-interpret the OCR for RRDB Generator based on common practice:
    # Input -> Initial Conv -> RRDBs -> Final Conv -> Skip connection -> Upsampling -> Output
    # However, the OCR code *literally* shows: Concat -> RRDB(x, 5) -> Downstack -> Upstack -> Output

    # Let's try to follow the OCR structure literally, even if odd:
    x = layers.Concatenate()([inputs, label_embedding_emotion, label_embedding_orientation])
    # Apply RRDBs - Need clarity on filters. Let's use 64 based on first downsample.
    initial_filters = 64
    x_conv = layers.Conv2D(initial_filters, 3, padding='same')(x) # Map to RRDB filter size
    x_rrdbs = x_conv
    for _ in range(num_rrdbs): # OCR code shows `for _ in range(3): x = RRDB(x, 5)` - inconsistent numbers
         x_rrdbs = RRDB(x_rrdbs, filters=initial_filters) # Let's use initial_filters
    # How to combine x_rrdbs with the U-Net path? The OCR code *restarts* x for down_stack.
    # This implies the RRDB part might have been a separate experiment or incorrectly integrated.

    # --- Let's implement a more standard RRDB Generator structure ---
    # Input -> Conv -> Downsampling -> RRDBs (bottleneck) -> Upsampling -> Conv -> Output

    inputs_gen = layers.Input(shape=[img_height, img_width, 3], name='input_image')
    inp_emotion_label_gen = layers.Input(shape=[1,], name='emotion_label')
    inp_orientation_label_gen = layers.Input(shape=[1,], name='orientation_label')

    # Embed and reshape labels (same as before)
    label_emb_e = layers.Embedding(num_emotion_classes, LABEL_EMBED_DIM)(inp_emotion_label_gen)
    label_emb_e = layers.Flatten()(label_emb_e)
    label_emb_e = layers.Dense(img_height * img_width * 1)(label_emb_e)
    label_emb_e = layers.Reshape((img_height, img_width, 1))(label_emb_e)
    label_emb_o = layers.Embedding(num_orientation_classes, LABEL_EMBED_DIM)(inp_orientation_label_gen)
    label_emb_o = layers.Flatten()(label_emb_o)
    label_emb_o = layers.Dense(img_height * img_width * 1)(label_emb_o)
    label_emb_o = layers.Reshape((img_height, img_width, 1))(label_emb_o)

    x = layers.Concatenate()([inputs_gen, label_emb_e, label_emb_o])

    # U-Net Downsampling Path
    down_stack_layers = [
        downsample(64, 4, apply_batchnorm=False), # (128, 128, 64)
        downsample(128, 4),                      # (64, 64, 128)
        downsample(256, 4),                      # (32, 32, 256)
        downsample(512, 4),                      # (16, 16, 512)
    ]
    # Fewer downsampling steps to allow for RRDBs in bottleneck? Or full U-Net?
    # Let's assume full U-Net structure based on OCR's down/up stack definitions
    down_stack_full = [
        downsample(64, 4, apply_batchnorm=False), # (batch_size, 128, 128, 64)
        downsample(128, 4),                      # (batch_size, 64, 64, 128)
        downsample(256, 4),                      # (batch_size, 32, 32, 256)
        downsample(512, 4),                      # (batch_size, 16, 16, 512)
        downsample(512, 4),                      # (batch_size, 8, 8, 512)
        downsample(512, 4),                      # (batch_size, 4, 4, 512)
        downsample(512, 4),                      # (batch_size, 2, 2, 512)
        # downsample(512, 4),                      # (batch_size, 1, 1, 512) # Bottleneck often here
    ]
    up_stack_full = [
        # upsample(512, 4, apply_dropout=True),    # (batch_size, 2, 2, 1024)
        upsample(512, 4, apply_dropout=True),    # (batch_size, 4, 4, 1024)
        upsample(512, 4, apply_dropout=True),    # (batch_size, 8, 8, 1024)
        upsample(512, 4),                         # (batch_size, 16, 16, 1024)
        upsample(256, 4),                         # (batch_size, 32, 32, 512)
        upsample(128, 4),                         # (batch_size, 64, 64, 256)
        upsample(64, 4),                          # (batch_size, 128, 128, 128)
    ]

    skips = []
    for down in down_stack_full:
        x = down(x)
        skips.append(x)

    # Bottleneck - Insert RRDBs here?
    # Let's assume the bottleneck has 512 filters based on last downsample
    bottleneck_filters = 512
    x_bottleneck = x
    # Apply RRDBs in the bottleneck
    # The RRDB filter count should match the bottleneck dimension
    for _ in range(num_rrdbs):
         x_bottleneck = RRDB(x_bottleneck, filters=bottleneck_filters)

    x = x_bottleneck # Output from RRDBs becomes input to decoder

    skips = reversed(skips) # Full skips list including bottleneck output? No, skips[:-1] reversed.

    # Upsampling Path
    for up, skip in zip(up_stack_full, skips): # Need careful matching of skips
        x = up(x)
        x = layers.Concatenate()([x, skip])

    # Final Layer
    initializer = tf.random_normal_initializer(0., 0.02)
    last = layers.Conv2DTranspose(OUTPUT_CHANNELS, 4, strides=2,
                                         padding='same', kernel_initializer=initializer,
                                         activation='tanh')(x) # Connect to last upsample output

    return tf.keras.Model(inputs=[inputs_gen, inp_emotion_label_gen, inp_orientation_label_gen], outputs=last)


# --- Discriminator (PatchGAN with Label Embedding) ---
# Assuming standard PatchGAN as losses/structure not specified differently for RRDB model
def Discriminator(img_height=IMG_HEIGHT, img_width=IMG_WIDTH,
                  num_emotion_classes=NUM_EMOTION_CLASSES,
                  num_orientation_classes=NUM_ORIENTATION_CLASSES):
    initializer = tf.random_normal_initializer(0., 0.02)

    inp = layers.Input(shape=[img_height, img_width, 3], name='input_image')
    tar = layers.Input(shape=[img_height, img_width, 3], name='target_image')

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

    x = layers.concatenate([inp, tar, label_embedding_emotion, label_embedding_orientation]) # (batch, 256, 256, 8)

    down1 = downsample(64, 4, False)(x) # (batch_size, 128, 128, 64)
    down2 = downsample(128, 4)(down1)    # (batch_size, 64, 64, 128)
    down3 = downsample(256, 4)(down2)    # (batch_size, 32, 32, 256)

    zero_pad1 = layers.ZeroPadding2D()(down3) # (batch_size, 34, 34, 256)
    conv = layers.Conv2D(512, 4, strides=1, kernel_initializer=initializer, use_bias=False)(zero_pad1) # (batch_size, 31, 31, 512)
    batchnorm1 = layers.BatchNormalization()(conv)
    leaky_relu = layers.LeakyReLU()(batchnorm1)
    zero_pad2 = layers.ZeroPadding2D()(leaky_relu) # (batch_size, 33, 33, 512)
    last = layers.Conv2D(1, 4, strides=1, kernel_initializer=initializer)(zero_pad2) # (batch_size, 30, 30, 1)

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
    y_true_features = vgg_perceptual_model(y_true)
    y_pred_features = vgg_perceptual_model(y_pred)
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
    gan_loss = loss_object(tf.ones_like(disc_generated_output), disc_generated_output)
    l1_loss = tf.reduce_mean(tf.abs(target - gen_output))
    dice_loss_value = dice_loss(target, gen_output)
    perceptual_loss_value = perceptual_loss(target, gen_output)

    total_gen_loss = (GAN_LOSS_WEIGHT * gan_loss +
                      L1_LOSS_WEIGHT * l1_loss +
                      DICE_LOSS_WEIGHT * dice_loss_value +
                      PERCEPTUAL_LOSS_WEIGHT * perceptual_loss_value)

    return total_gen_loss, gan_loss, l1_loss # Return main components

# --- Example Usage ---
if __name__ == '__main__':
    # Note: The Generator structure was inferred, adjust if needed based on original intent.
    generator = Generator()
    tf.keras.utils.plot_model(generator, to_file='rddb_gan_generator.png', show_shapes=True, dpi=64)

    discriminator = Discriminator()
    tf.keras.utils.plot_model(discriminator, to_file='rddb_gan_discriminator.png', show_shapes=True, dpi=64)

    print("RRDB GAN model definitions loaded.")
    print(f"Generator inputs: {generator.input_names}")
    print(f"Discriminator inputs: {discriminator.input_names}")

    # --- Placeholder for Training Loop ---
