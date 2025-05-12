# GAN for Facial Expression and Orientation Manipulation

This repository contains the implementation of various Generative Adversarial Network (GAN) architectures explored for the task of manipulating facial expressions and orientations, based on the IIITM Face Emotion dataset.

## Objective

To create GAN models capable of taking a facial image and target attributes (expression and orientation) as input, and generating a new image reflecting those attributes while preserving the subject's identity.

## Dataset

*   **Name:** IIITM Face Emotion Dataset (modified from IIITM Face Data)
*   **Description:** Contains 1928 images from 107 participants (87 male, 20 female). Images feature 6 facial expressions (Smile, Surprise, Surprise w/ Mouth Open, Neutral, Sad, Yawning) across 3 vertical orientations (Front, Up, Down). Facial regions are segmented and resized to 800x1000 pixels (aspect ratio 4:5), though the models here typically use 256x256 inputs.
*   **Nomenclature:** `SUB XX EE O` (XX: Subject ID, EE: Emotion Code, O: Orientation Code)
    *   Emotions: SM, SU, SO, NE, SA, YN
    *   Orientations: F, D, U

## Model Architectures Implemented

This repository includes Python scripts for the following model variations explored:

1.  **`modified_pix2pix.py`**: A modified Pix2pix architecture incorporating label embeddings for expression and orientation, concatenated to the input image before the U-Net encoder. Uses GAN + L1 loss.
2.  **`native_pix2pix.py`**: A standard Pix2pix U-Net architecture without explicit label embedding inputs. Uses GAN + L1 + Dice + Perceptual (VGG) loss. ( PSNR 7).
3.  **`efficient_vit.py`**: Replaces the CNN encoder/decoder with an EfficientViT transformer. Includes label embeddings concatenated to the input. Uses GAN + L1 + Dice + Perceptual + FID loss. ( PSNR 14.5).
4.  **`rddb_gan.py`**: Uses Residual-in-Residual Dense Blocks (RRDB) within a U-Net structure, inspired by ESRGAN. Includes label embeddings. Uses GAN + L1 + Dice + Perceptual loss. (Reported 10k steps, PSNR 18).
5.  **`rddb_esrgan.py`**: Builds on `rddb_gan.py` by adding a pre-trained ESRGAN model (from TF Hub) at the end of the generator for potential quality enhancement/upscaling. Uses GAN + L1 + Dice + Perceptual loss. (PSNR 16).
6.  **`vgg16_rddb_esrgan.py`**: Incorporates features extracted by a pre-trained VGG16 early in the generator, concatenated with the main feature stream. Uses RRDB blocks and the final ESRGAN TF Hub layer. Includes edge loss. ( PSNR 20).
7.  **`vgg16_rddb.py`**: Similar to the previous model but removes the final ESRGAN TF Hub layer. Includes edge loss. ( PSNR 27).

Here is the updated table with the PSNR values included:

| **Method**                | **PSNR (dB)** | **Remarks**                                                                                     |
|---------------------------|---------------|-------------------------------------------------------------------------------------------------|
| `modified_pix2pix.py`     | N/A           | Modified Pix2pix with label embeddings and U-Net encoder.                                      |
| `native_pix2pix.py`       | 7.0           | Standard Pix2pix U-Net architecture.                                                          |
| `efficient_vit.py`        | 14.5          | EfficientViT transformer-based architecture with label embeddings.                             |
| `rddb_gan.py`            | 18.0          | RRDB-based U-Net structure inspired by ESRGAN.                                                |
| `rddb_esrgan.py`         | 16.0          | Adds pre-trained ESRGAN for quality enhancement/upscaling.                                     |
| `vgg16_rddb_esrgan.py`   | 20.0          | Incorporates pre-trained VGG16 features and RRDB blocks with ESRGAN.                          |
| `vgg16_rddb.py`          | 27.0          | RRDB blocks and pre-trained VGG16 features but excludes ESRGAN layer, includes edge loss.      |



## Setup

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/<your-username>/Facial-Expression-GAN.git
    cd Facial-Expression-GAN
    ```
2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```
3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
4.  **Download the Dataset:**
    Download the IIITM Face Emotion dataset. You will need to adapt the data loading part of the scripts to point to this directory and parse the filenames/labels.

## Usage (Conceptual)

Each `.py` file defines the Generator, Discriminator, and associated loss functions for a specific architecture. You would typically need a main training script (not provided here, but structure is implied within the files) to:

1.  Load and preprocess the dataset (images and labels).
2.  Instantiate the Generator and Discriminator from the desired model file (e.g., `from rddb_gan import Generator, Discriminator`).
3.  Define optimizers (e.g., Adam).
4.  Implement the training step function (calculating losses, applying gradients).
5.  Run the training loop for a specified number of epochs or steps.
6.  Implement functions for generating images using the trained generator and evaluating metrics.
