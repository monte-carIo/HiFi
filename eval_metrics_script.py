"""
Called like so:
python eval_metrics_script.py --generated_views=/root/workspace/gaussian-splatting/output/10/test/ours_30000/renders \
    --ground_truth_views=/root/workspace/data/spinnerf-dataset/10/images_4
"""
import os
import glob
from absl import app
from absl import flags

import imageio
import numpy as np
import logging
import tensorflow as tf
from scipy.linalg import sqrtm

import eval_utils

FLAGS = flags.FLAGS

flags.DEFINE_string('generated_views', '', 'Directory to generated views.')
flags.DEFINE_string('ground_truth_views', '', 'Directory to ground truth views.')

# Configure logging
log_file = "evaluation_metrics.log"
logging.basicConfig(
    filename=log_file,
    filemode='a',  # Append mode
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

# Function to load the InceptionV3 model for FID
def load_inception_v3():
    """Loads the InceptionV3 model for feature extraction."""
    base_model = tf.keras.applications.InceptionV3(
        include_top=False,
        pooling='avg',
        input_shape=(299, 299, 3)
    )
    return base_model

# Function to calculate FID
def calculate_fid(features1, features2):
    """Compute the Fréchet Inception Distance."""
    # Calculate means and covariances
    mu1, sigma1 = np.mean(features1, axis=0), np.cov(features1, rowvar=False)
    mu2, sigma2 = np.mean(features2, axis=0), np.cov(features2, rowvar=False)

    # Compute squared difference of means
    diff = mu1 - mu2
    mean_diff = diff.dot(diff)

    # Compute product of covariances
    cov_mean, _ = sqrtm(sigma1.dot(sigma2), disp=False)

    # Handle numerical instability
    if np.iscomplexobj(cov_mean):
        cov_mean = cov_mean.real

    return mean_diff + np.trace(sigma1 + sigma2 - 2 * cov_mean)

def preprocess_for_inception(img):
    """Preprocesses an image for InceptionV3 feature extraction."""
    img = tf.image.resize(img, (299, 299))
    img = tf.keras.applications.inception_v3.preprocess_input(img)
    return img

def main(argv):
    if len(argv) > 1:
        raise app.UsageError('Too many command-line arguments.')

    # Load models
    lpips_fn = eval_utils.load_lpips()
    inception_model = load_inception_v3()

    mse_fn = lambda x, y: np.mean((x - y) ** 2)
    psnr_fn = lambda x, y: -10 * np.log10(mse_fn(x, y))

    def ComputeMetrics(generated, gt):
        ssim_score = eval_utils.ssim(generated, gt)
        float32_gen = (generated / 255.).astype(np.float32)
        float32_gt = (gt / 255.).astype(np.float32)
        lpips_score = lpips_fn(float32_gen, float32_gt)
        psnr_score = psnr_fn(float32_gen, float32_gt)
        return ssim_score, psnr_score, lpips_score

    images_to_eval = glob.glob(os.path.join(FLAGS.generated_views, "*.png")) \
                    + glob.glob(os.path.join(FLAGS.generated_views, "*.jpg"))
    files = [os.path.basename(s) for s in images_to_eval]
    images_gt = glob.glob(os.path.join(FLAGS.ground_truth_views, "*.png")) \
                    + glob.glob(os.path.join(FLAGS.ground_truth_views, "*.jpg"))
    files_gt = sorted([os.path.basename(s) for s in images_gt])[:40]
    

    ssim = []
    psnr = []
    lpips = []

    inception_features_generated = []
    inception_features_gt = []

    for k, gt in zip(files, files_gt):
        try:
            gt_im = imageio.imread(os.path.join(FLAGS.ground_truth_views, gt))
            gv_im = imageio.imread(os.path.join(FLAGS.generated_views, k))
            if gv_im.shape != gt_im.shape:
                gv_im = tf.image.resize(gv_im, gt_im.shape[:2]).numpy()
        except Exception as e:
            logging.error(f"I/O Error opening filename: {k}. Error: {e}")
            continue

        # Compute SSIM, PSNR, LPIPS
        ssim_score, psnr_score, lpips_score = ComputeMetrics(gt_im, gv_im)
        ssim.append(ssim_score)
        psnr.append(psnr_score)
        lpips.append(lpips_score)

        # Preprocess for Inception and extract features
        gt_preprocessed = preprocess_for_inception(gt_im).numpy()
        gv_preprocessed = preprocess_for_inception(gv_im).numpy()

        gt_features = inception_model(tf.expand_dims(gt_preprocessed, 0)).numpy()
        gv_features = inception_model(tf.expand_dims(gv_preprocessed, 0)).numpy()

        inception_features_gt.append(gt_features)
        inception_features_generated.append(gv_features)

    # Compute FID
    inception_features_gt = np.vstack(inception_features_gt)
    inception_features_generated = np.vstack(inception_features_generated)
    fid_score = calculate_fid(inception_features_generated, inception_features_gt)

    # Log and print the results
    metrics_log = (
        f"data: {FLAGS.generated_views}\n\n"
        f"PSNR:\nMean: {np.mean(psnr):.4f}\nStddev: {np.std(psnr):.4f}\n\n"
        f"SSIM:\nMean: {np.mean(ssim):.4f}\nStddev: {np.std(ssim):.4f}\n\n"
        f"LPIPS:\nMean: {np.mean(lpips):.4f}\nStddev: {np.std(lpips):.4f}\n\n"
        f"FID:\nScore: {fid_score:.4f}\n"
    )

    print(metrics_log)
    logging.info(metrics_log)

if __name__ == '__main__':
    app.run(main)
