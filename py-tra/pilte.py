from PIL import Image
import tifffile as tf
import numpy as np
import os
from methods.Brovey import Brovey
from methods.PCA import PCA
from methods.IHS import IHS
from methods.SFIM import SFIM
from methods.GS import GS
from methods.GFPCA import GFPCA

ms = tf.imread(r"D:\pans\pan-sharpening\fullresolution\LR000_normal.tiff")
print(ms.shape)
pan = tf.imread(r"D:\pans\pan-sharpening\fullresolution\pan.tiff")
print(np.min(ms))

used_pan = np.expand_dims(pan, -1)/255
used_ms = ms
save_dir = 'output'
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
fused_image = Brovey(used_pan[:, :, :], used_ms[:, :, :])
tf.imsave("output/brovey.tiff", fused_image)

fused_image = PCA(used_pan[:, :, :], used_ms[:, :, :])
tf.imsave("output/pca.tiff", fused_image)

fused_image = IHS(used_pan[:, :, :], used_ms[:, :, :])
tf.imsave("output/IHS.tiff", fused_image)

fused_image = SFIM(used_pan[:, :, :], used_ms[:, :, :])
tf.imsave("output/SFIM.tiff", fused_image)

fused_image = GS(used_pan[:, :, :], used_ms[:, :, :])
tf.imsave("output/GS.tiff", fused_image)

fused_image = GFPCA(used_pan[:, :, :], used_ms[:, :, :])
tf.imsave("output/GFPCA.tiff", fused_image)
