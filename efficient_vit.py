import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.models import Model
import tensorflow_probability as tfp

# --- Configuration ---
OUTPUT_CHANNELS = 3
IMG_WIDTH = 256
IMG_HEIGHT = 256
PATCH_SIZE = 16
NUM_PATCHES = (IMG_HEIGHT // PATCH_SIZE) ** 2
PROJECTION_DIM = 64 # Example value, adjust as needed
NUM_HEADS = 4 # Example value
TRANSFORMER_LAYERS = 4 # Example value
MLP_DIM = 128 # Example value

# Label embedding dims
NUM_EMOTION_CLASSES = 6 
NUM_ORIENTATION_CLASSES = 3 
LABEL_EMBED_DIM = 256

GAN_LOSS_WEIGHT = 100
L1_LOSS_WEIGHT = 50
DICE_LOSS_WEIGHT = 70
PERCEPTUAL_LOSS_WEIGHT = 150
FID_LOSS_WEIGHT = 1 # Weight for FID loss component

# --- Vision Transformer Blocks ---

class PatchEmbedding(layers.Layer):
    def __init__(self, patch_size, num_patches, projection_dim):
        super(PatchEmbedding, self).__init__()
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.projection = layers.Dense(units=projection_dim)

    def build(self, input_shape):
        # This method ensures that the projection layer has a defined input shape
        # Input shape to projection: (None, num_patches, patch_h * patch_w * channels)
        # Calculate the size of the flattened patch vector
        # Note: input_shape[-1] here is the channel dim *after* concatenation with labels
        patch_vec_size = self.patch_size * self.patch_size * input_shape[-1]
        self.projection.build((None, self.num_patches, patch_vec_size))
        super(PatchEmbedding, self).build(input_shape) # Call the parent build method

    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding='VALID')
        # Reshape patches: (batch_size, num_patches_h, num_patches_w, patch_h*patch_w*channels)
        # -> (batch_size, num_patches, patch_h*patch_w*channels)
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, self.num_patches, patch_dims])
        return self.projection(patches)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.num_patches, self.projection.units)

class EfficientViTBlock(layers.Layer):
    def __init__(self, projection_dim, num_heads, mlp_dim, dropout=0.1):
        super(EfficientViTBlock, self).__init__()
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.mha = layers.MultiHeadAttention(num_heads=num_heads, key_dim=projection_dim)
        self.dropout1 = layers.Dropout(dropout)

        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.mlp = tf.keras.Sequential([
            layers.Dense(mlp_dim, activation=tf.nn.gelu),
            layers.Dropout(dropout),
            layers.Dense(projection_dim),
            layers.Dropout(dropout),
        ])

    def call(self, x):
        # First part: multi-head self-attention
        residual1 = x
        x_ln1 = self.layernorm1(x)
        attn_output = self.mha(query=x_ln1, value=x_ln1, key=x_ln1) # Self-attention
        attn_output = self.dropout1(attn_output)
        out1 = attn_output + residual1 # Residual connection

        # Second part: MLP block
        residual2 = out1
        out1_ln2 = self.layernorm2(out1)
        mlp_output = self.mlp(out1_ln2)
        out2 = mlp_output + residual2 # Residual connection
        return out2

def create_vit_encoder(input_shape, num_patches, projection_dim, num_heads, transformer_layers, mlp_dim):
    inputs = layers.Input(shape=input_shape)
    # Create patches.
    patches = PatchEmbedding(PATCH_SIZE, num_patches, projection_dim)(inputs)
    # Create multiple layers of the Transformer block.
    encoded_patches = patches
    for _ in range(transformer_layers):
        encoded_patches = EfficientViTBlock(projection_dim, num_heads, mlp_dim)(encoded_patches)

    # Reshape output back to image-like shape for decoder - Adjust target shape as needed
    # Output shape before reshape: (batch, num_patches, projection_dim)
    # Example: Reshape to (batch, sqrt(num_patches), sqrt(num_patches), projection_dim)
    feature_map_size = int(num_patches**0.5)
    output = layers.Reshape((feature_map_size, feature_map_size, projection_dim))(encoded_patches)
    return tf.keras.Model(inputs=inputs, outputs=output, name="vit_encoder")


# --- Building Blocks (Modified for EfficientViT) ---
# Using LayerNorm and Swish/PReLU as seen in the Discriminator description for this model

def downsample_vit(filters, size, apply_layernorm=True): # Changed apply_batchnorm to apply_layernorm
    initializer = tf.random_normal_initializer(0., 0.02)
    result = tf.keras.Sequential()
    result.add(
        layers.Conv2D(filters, size, strides=2, padding='same',
                             kernel_initializer=initializer, use_bias=False))
    if apply_layernorm:
        result.add(layers.LayerNormalization()) # Changed to LayerNormalization
    result.add(layers.PReLU(shared_axes=[1, 2])) # Changed to PReLU from LeakyReLU based on Disc code
    return result

def upsample_vit(filters, size, apply_dropout=False):
    initializer = tf.random_normal_initializer(0., 0.02)
    result = tf.keras.Sequential()
    result.add(
        layers.Conv2DTranspose(filters, size, strides=2,
                                        padding='same',
                                        kernel_initializer=initializer,
                                        use_bias=False))
    result.add(layers.LayerNormalization()) # Changed to LayerNormalization
    if apply_dropout:
        result.add(layers.Dropout(0.5))
    result.add(layers.Activation('swish')) # Changed to Swish from ReLU based on Disc code
    return result


# --- EfficientViT Generator Model ---
def EfficientViT_Generator(img_height=IMG_HEIGHT, img_width=IMG_WIDTH,
                           num_emotion_classes=NUM_EMOTION_CLASSES,
                           num_orientation_classes=NUM_ORIENTATION_CLASSES):

    inputs = layers.Input(shape=[img_height, img_width, 3], name='input_image')

    # Emotion Label Input and Embedding
    inp_emotion_label = layers.Input(shape=[1,], name='emotion_label')
    label_embedding_emotion = layers.Embedding(num_emotion_classes, LABEL_EMBED_DIM)(inp_emotion_label)
    label_embedding_emotion = layers.Flatten()(label_embedding_emotion)
    label_embedding_emotion = layers.Dense(img_height * img_width * 1)(label_embedding_emotion)
    label_embedding_emotion = layers.Reshape((img_height, img_width, 1))(label_embedding_emotion)

    # Orientation Label Input and Embedding
    inp_orientation_label = layers.Input(shape=[1,], name='orientation_label')
    label_embedding_orientation = layers.Embedding(num_orientation_classes, LABEL_EMBED_DIM)(inp_orientation_label)
    label_embedding_orientation = layers.Flatten()(label_embedding_orientation)
    label_embedding_orientation = layers.Dense(img_height * img_width * 1)(label_embedding_orientation)
    label_embedding_orientation = layers.Reshape((img_height, img_width, 1))(label_embedding_orientation)

    # Concatenate inputs and embeddings
    # Input shape to ViT: (batch, H, W, C + Emb1 + Emb2)
    combined_input = layers.Concatenate(axis=-1)([inputs, label_embedding_emotion, label_embedding_orientation])
    combined_input_shape = combined_input.shape[1:] # H, W, C_combined

    # ViT Model replacing the original CNN-based encoder-decoder
    vit_encoder = create_vit_encoder(
        input_shape=combined_input_shape,
        num_patches=NUM_PATCHES,
        projection_dim=PROJECTION_DIM,
        num_heads=NUM_HEADS,
        transformer_layers=TRANSFORMER_LAYERS,
        mlp_dim=MLP_DIM
    )

    x = vit_encoder(combined_input) # Output shape (batch, 16, 16, projection_dim)

    # Upsample path (CNN based decoder, potentially could be ViT based too)
    # Needs to upsample from (16, 16, projection_dim) to (256, 256, 3)
    # The number of upsampling layers needs to match the downsampling in ViT patch extraction (256 -> 16 is 4 steps of stride 2)
    x = upsample_vit(128, 4)(x) # (batch, 32, 32, 128) - Adjust filter counts as needed
    x = upsample_vit(64, 4)(x)  # (batch, 64, 64, 64)
    x = upsample_vit(32, 4)(x)  # (batch, 128, 128, 32)

    initializer = tf.random_normal_initializer(0., 0.02)
    last = layers.Conv2DTranspose(OUTPUT_CHANNELS, 4, strides=2,
                                         padding='same', kernel_initializer=initializer,
                                         activation='tanh')(x) # (batch_size, 256, 256, 3)

    return tf.keras.Model(inputs=[inputs, inp_emotion_label, inp_orientation_label], outputs=last)

# --- Discriminator (PatchGAN with Label Embedding and ViT-style layers) ---

def Discriminator(img_height=IMG_HEIGHT, img_width=IMG_WIDTH,
                  num_emotion_classes=NUM_EMOTION_CLASSES,
                  num_orientation_classes=NUM_ORIENTATION_CLASSES):
    initializer = tf.random_normal_initializer(0., 0.02)

    inp = layers.Input(shape=[img_height, img_width, 3], name='input_image')
    tar = layers.Input(shape=[img_height, img_width, 3], name='target_image')

    # Emotion Label Input and Embedding
    inp_emotion_label = layers.Input(shape=[1,], name='emotion_label')
    label_embedding_emotion = layers.Embedding(num_emotion_classes, LABEL_EMBED_DIM)(inp_emotion_label)
    label_embedding_emotion = layers.Flatten()(label_embedding_emotion)
    label_embedding_emotion = layers.Dense(img_height * img_width * 1)(label_embedding_emotion)
    label_embedding_emotion = layers.Reshape((img_height, img_width, 1))(label_embedding_emotion)

    # Orientation Label Input and Embedding
    inp_orientation_label = layers.Input(shape=[1,], name='orientation_label')
    label_embedding_orientation = layers.Embedding(num_orientation_classes, LABEL_EMBED_DIM)(inp_orientation_label)
    label_embedding_orientation = layers.Flatten()(label_embedding_orientation)
    label_embedding_orientation = layers.Dense(img_height * img_width * 1)(label_embedding_orientation)
    label_embedding_orientation = layers.Reshape((img_height, img_width, 1))(label_embedding_orientation)

    # Concatenate inputs, target, and embeddings
    x = layers.concatenate([inp, tar, label_embedding_emotion, label_embedding_orientation]) # (batch, 256, 256, 3+3+1+1=8)

    # Using downsample_vit which uses LayerNorm and PReLU
    down1 = downsample_vit(64, 4, False)(x)      # (batch_size, 128, 128, 64)
    down2 = downsample_vit(128, 4)(down1)         # (batch_size, 64, 64, 128)
    down3 = downsample_vit(256, 4)(down2)         # (batch_size, 32, 32, 256)

    zero_pad1 = layers.ZeroPadding2D()(down3)     # (batch_size, 34, 34, 256)
    conv = layers.Conv2D(512, 4, strides=1,
                                kernel_initializer=initializer,
                                use_bias=False)(zero_pad1) # (batch_size, 31, 31, 512)

    norm1 = layers.LayerNormalization()(conv)       # Changed to LayerNormalization
    leaky_relu = layers.PReLU(shared_axes=[1, 2])(norm1) # Changed to PReLU

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

# --- VGG Perceptual Loss --- (Same as native_pix2pix)
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

# --- Dice Loss --- (Same as native_pix2pix)
def dice_loss(y_true, y_pred, smooth=1e-5):
    intersection = tf.reduce_sum(y_true * y_pred, axis=[1, 2, 3])
    union = tf.reduce_sum(y_true, axis=[1, 2, 3]) + tf.reduce_sum(y_pred, axis=[1, 2, 3])
    dice = (2. * intersection + smooth) / (union + smooth)
    return 1.0 - tf.reduce_mean(dice)

# --- FID Loss ---
# Using InceptionV3 for FID calculation as per OCR
inception_model_fid = InceptionV3(include_top=False, pooling='avg', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3))

def extract_features(image):
    # Preprocess for InceptionV3 if needed (e.g., scale to [0,1] or use preprocess_input)
    # Assuming image is already in [-1, 1], scale to [0, 1] for this example
    image_01 = (image + 1.0) / 2.0
    # image_preprocessed = tf.keras.applications.inception_v3.preprocess_input(image_01 * 255.0) # Alternative
    features = inception_model_fid(image_01)
    return features

# Note: Calculating FID requires statistics over a larger set of images,
# doing it per batch within the loss function is an approximation (sometimes called batch FID).
# A proper FID calculation should be done offline or accumulated over many batches.
# This fid_loss function calculates a distance based on current batch stats.
def fid_loss(generated_image, real_image):
    real_image = tf.cast(real_image, tf.float32)
    generated_image = tf.cast(generated_image, tf.float32)

    real_features = extract_features(real_image)
    generated_features = extract_features(generated_image)

    # Calculate mean and covariance statistics from the batch features
    real_mean = tf.reduce_mean(real_features, axis=0)
    # Use sample_axis=0 for batch covariance
    real_cov = tfp.stats.covariance(real_features, sample_axis=0)
    generated_mean = tf.reduce_mean(generated_features, axis=0)
    generated_cov = tfp.stats.covariance(generated_features, sample_axis=0)

    # Calculate FID-like distance for the batch
    # ||mu_real - mu_gen||^2
    mean_diff_sq = tf.reduce_sum(tf.square(real_mean - generated_mean))
    # Trace(cov_real + cov_gen - 2 * sqrt(cov_real * cov_gen)) - Needs matrix sqrt
    # Simpler approximation used in OCR: Trace(covariance_difference)
    # covariance_difference = real_cov - generated_cov # This isn't quite right for FID
    # trace_term = tf.linalg.trace(covariance_difference) # OCR uses this simpler trace

    # A more direct Frechet distance component (without matrix sqrt complexity):
    # Froebenius norm of covariance difference: tf.norm(real_cov - generated_cov, ord='fro', axis=[-2,-1])
    # Let's use the simple version from OCR:
    trace_term = tf.linalg.trace(real_cov - generated_cov) # Approximation
    fid_value = mean_diff_sq + trace_term

    # Ensure non-negative, return as tensor
    return tf.nn.relu(tf.cast(fid_value, tf.float32))


def generator_loss(disc_generated_output, gen_output, target):
    gan_loss = loss_object(tf.ones_like(disc_generated_output), disc_generated_output)
    l1_loss = tf.reduce_mean(tf.abs(target - gen_output))
    dice_loss_value = dice_loss(target, gen_output)
    perceptual_loss_value = perceptual_loss(target, gen_output)
    fid_loss_value = fid_loss(gen_output, target) # Calculate FID-like batch loss

    # Combine losses with weights from OCR
    total_gen_loss = (GAN_LOSS_WEIGHT * gan_loss +
                      L1_LOSS_WEIGHT * l1_loss +
                      DICE_LOSS_WEIGHT * dice_loss_value +
                      PERCEPTUAL_LOSS_WEIGHT * perceptual_loss_value +
                      FID_LOSS_WEIGHT * fid_loss_value)

    return total_gen_loss, gan_loss, l1_loss, dice_loss_value, perceptual_loss_value, fid_loss_value

# --- Example Usage ---
if __name__ == '__main__':
    generator = EfficientViT_Generator()
    # generator.summary() # May be very long due to ViT blocks
    tf.keras.utils.plot_model(generator, to_file='efficient_vit_generator.png', show_shapes=True, dpi=64)

    discriminator = Discriminator()
    # discriminator.summary()
    tf.keras.utils.plot_model(discriminator, to_file='efficient_vit_discriminator.png', show_shapes=True, dpi=64)

    print("EfficientViT model definitions loaded.")
    print(f"Generator inputs: {generator.input_names}")
    print(f"Discriminator inputs: {discriminator.input_names}")

    # --- Placeholder for Training Loop ---
    # Similar to modified_pix2pix.py, requires images and labels.
