import tensorflow as tf
from tensorflow.keras import layers

# --- Configuration ---
OUTPUT_CHANNELS = 3
LAMBDA = 100 # Weight for L1 loss
IMG_WIDTH = 256 # Models seem to use 256x256 input
IMG_HEIGHT = 256
NUM_EMOTION_CLASSES = 6 # SM, SU, SO, NE, SA, YN
NUM_ORIENTATION_CLASSES = 3 # F, D, U

# --- Building Blocks ---

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

# --- Generator (Modified U-Net with Label Embedding) ---

def Generator(img_height=IMG_HEIGHT, img_width=IMG_WIDTH,
              num_emotion_classes=NUM_EMOTION_CLASSES,
              num_orientation_classes=NUM_ORIENTATION_CLASSES):
    inputs = layers.Input(shape=[img_height, img_width, 3], name='input_image')

    # Emotion Label Input and Embedding
    inp_emotion_label = layers.Input(shape=[1,], name='emotion_label')
    label_embedding_emotion = layers.Embedding(num_emotion_classes, 256)(inp_emotion_label)
    label_embedding_emotion = layers.Flatten()(label_embedding_emotion)
    label_embedding_emotion = layers.Dense(img_height * img_width * 1)(label_embedding_emotion) # Project to image size
    label_embedding_emotion = layers.Reshape((img_height, img_width, 1))(label_embedding_emotion)

    # Orientation Label Input and Embedding
    inp_orientation_label = layers.Input(shape=[1,], name='orientation_label')
    label_embedding_orientation = layers.Embedding(num_orientation_classes, 256)(inp_orientation_label)
    label_embedding_orientation = layers.Flatten()(label_embedding_orientation)
    label_embedding_orientation = layers.Dense(img_height * img_width * 1)(label_embedding_orientation) # Project to image size
    label_embedding_orientation = layers.Reshape((img_height, img_width, 1))(label_embedding_orientation)

    # Concatenate inputs and embeddings
    x = layers.Concatenate()([inputs, label_embedding_emotion, label_embedding_orientation]) # Shape: (batch, 256, 256, 3+1+1=5)

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
    last = layers.Conv2DTranspose(OUTPUT_CHANNELS, 4,
                                         strides=2,
                                         padding='same',
                                         kernel_initializer=initializer,
                                         activation='tanh') # (batch_size, 256, 256, 3)

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

    return tf.keras.Model(inputs=[inputs, inp_emotion_label, inp_orientation_label], outputs=x)

# --- Discriminator (PatchGAN with Label Embedding) ---

def Discriminator(img_height=IMG_HEIGHT, img_width=IMG_WIDTH,
                  num_emotion_classes=NUM_EMOTION_CLASSES,
                  num_orientation_classes=NUM_ORIENTATION_CLASSES):
    initializer = tf.random_normal_initializer(0., 0.02)

    inp = layers.Input(shape=[img_height, img_width, 3], name='input_image')
    tar = layers.Input(shape=[img_height, img_width, 3], name='target_image')

    # Emotion Label Input and Embedding
    inp_emotion_label = layers.Input(shape=[1,], name='emotion_label')
    label_embedding_emotion = layers.Embedding(num_emotion_classes, 256)(inp_emotion_label)
    label_embedding_emotion = layers.Flatten()(label_embedding_emotion)
    label_embedding_emotion = layers.Dense(img_height * img_width * 1)(label_embedding_emotion)
    label_embedding_emotion = layers.Reshape((img_height, img_width, 1))(label_embedding_emotion)

    # Orientation Label Input and Embedding
    inp_orientation_label = layers.Input(shape=[1,], name='orientation_label')
    label_embedding_orientation = layers.Embedding(num_orientation_classes, 256)(inp_orientation_label)
    label_embedding_orientation = layers.Flatten()(label_embedding_orientation)
    label_embedding_orientation = layers.Dense(img_height * img_width * 1)(label_embedding_orientation)
    label_embedding_orientation = layers.Reshape((img_height, img_width, 1))(label_embedding_orientation)

    # Concatenate inputs, target, and embeddings
    x = layers.concatenate([inp, tar, label_embedding_emotion, label_embedding_orientation]) # (batch, 256, 256, 3+3+1+1=8)

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

    return tf.keras.Model(inputs=[inp, tar, inp_emotion_label, inp_orientation_label], outputs=last)

# --- Loss Functions ---
loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def discriminator_loss(disc_real_output, disc_generated_output):
    real_loss = loss_object(tf.ones_like(disc_real_output), disc_real_output)
    generated_loss = loss_object(tf.zeros_like(disc_generated_output), disc_generated_output)
    total_disc_loss = real_loss + generated_loss
    return total_disc_loss

def generator_loss(disc_generated_output, gen_output, target):
    gan_loss = loss_object(tf.ones_like(disc_generated_output), disc_generated_output)
    # Mean absolute error (L1 Loss)
    l1_loss = tf.reduce_mean(tf.abs(target - gen_output))
    total_gen_loss = gan_loss + (LAMBDA * l1_loss)
    return total_gen_loss, gan_loss, l1_loss

# --- Example Usage (Illustrative) ---
if __name__ == '__main__':
    generator = Generator()
    # generator.summary()
    tf.keras.utils.plot_model(generator, to_file='modified_pix2pix_generator.png', show_shapes=True, dpi=64)

    discriminator = Discriminator()
    # discriminator.summary()
    tf.keras.utils.plot_model(discriminator, to_file='modified_pix2pix_discriminator.png', show_shapes=True, dpi=64)

    print("Modified Pix2pix model definitions loaded.")
    print(f"Generator inputs: {generator.input_names}")
    print(f"Discriminator inputs: {discriminator.input_names}")

    # --- Placeholder for Training Loop ---
    # 1. Load and preprocess dataset (images, emotion labels, orientation labels)
    # 2. Create tf.data.Dataset objects
    # 3. Define optimizers (e.g., Adam)
    # 4. Instantiate Generator and Discriminator
    # 5. Define checkpointing
    # 6. Implement the train_step function:
    #    - Get batch of inputs, target images, emotion labels, orientation labels
    #    - Use tf.GradientTape to compute gradients for generator and discriminator
    #    - Calculate generator loss (gan_loss + LAMBDA * l1_loss)
    #    - Calculate discriminator loss
    #    - Apply gradients using optimizers
    # 7. Run the training loop
    # 8. Periodically save checkpoints and generate sample images
