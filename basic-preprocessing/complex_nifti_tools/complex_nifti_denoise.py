# -*- coding: utf-8 -*-
import numpy as np
import ants
import argparse
import sys
from scipy.ndimage import gaussian_filter
from skimage.restoration import unwrap_phase

__author__ = "Ruediger Stirnberg"
__email__ = "ruediger.stirnberg@dzne.de"

TAG = '[CNLM]'

def complex_antsnlm_denoise(magn_file, phas_file, phase_scale, out=None, diff=False, background_free=False):

    print(TAG, 'Load magnitude')
    magn = ants.image_read(magn_file)
    temp3D = ants.image_read(magn_file, dimension=3)
    print(TAG, 'Load phase')
    phas = ants.image_read(phas_file)

    comp = magn.numpy().astype(np.float32) * np.exp(1.0j*phas.numpy().astype(np.float32)*phase_scale)

    if out is None:
        magn_out_file = magn_file + '.nlm.nii.gz'
        phas_out_file = phas_file + '.nlm.nii.gz'
        magn_diff_file = magn_file + '.noise.nii.gz'
        phas_diff_file = phas_file + '.noise.nii.gz'
    else:
        magn_out_file = out + '_magn_nlm.nii.gz'
        phas_out_file = out + '_phas_nlm.nii.gz'
        magn_diff_file = out + '_magn_noise.nii.gz'
        phas_diff_file = out + '_phas_noise.nii.gz'

    if len(comp.shape)==3:
        comp = comp[...,np.newaxis]
    

    print(TAG, 'Get a mask')
    mask = ants.get_mask(temp3D, low_thresh=temp3D.mean()/5.0) #include even dark regions, but not all voxels outside the brain (it just takes too long to process)

    axis=3
    N = comp.shape[axis]
    slc = [slice(None)]*len(comp.shape)
    comp_denoise = np.zeros_like(comp)
    for n in range(N):
        slc[axis] = n

        if not background_free:
            print(TAG, 'Compute unwrapped and smoothed background phase approximation')
            phas_baseline = unwrap_phase(np.angle(comp[tuple(slc)]))
            phas_baseline = gaussian_filter(phas_baseline,4)
        else:
            phas_baseline = 0.0

        print(TAG, 'Temporarily compute real and imaginary images')
        real = np.real(comp[tuple(slc)] * np.exp(-1.0j*phas_baseline))/2.0+4096/2
        imag = np.imag(comp[tuple(slc)] * np.exp(-1.0j*phas_baseline))/2.0+4096/2
        real = temp3D.new_image_like(real)
        imag = temp3D.new_image_like(imag)


        print(TAG, "Denoise temporary real image")
        real_denoise = ants.denoise_image(real, mask, noise_model='Gaussian')
        real_denoise = real_denoise*2-4096

        print(TAG, "Denoise temporary imaginary image")
        imag_denoise = ants.denoise_image(imag, mask, noise_model='Gaussian')
        imag_denoise = imag_denoise*2-4096

        del real, imag

        print(TAG, "Convert back to denoised magnitude and phase")
        comp_denoise[tuple(slc)] = (real_denoise.numpy()) + 1.0j*(imag_denoise.numpy())
        comp_denoise[tuple(slc)] *= np.exp(1.0j*phas_baseline)

    magn_denoise = magn.new_image_like(np.squeeze(np.abs(comp_denoise)))
    phas_denoise = phas.new_image_like(np.squeeze(np.angle(comp_denoise))/phase_scale)
    
    print(TAG, 'Save magnitude', magn_out_file)
    ants.image_write(magn_denoise, magn_out_file)

    print(TAG, 'Save phase', phas_out_file)
    ants.image_write(phas_denoise, phas_out_file)

    if diff:
        comp_noise = comp - comp_denoise
        magn_noise = magn.new_image_like(np.abs(comp_noise))
        phas_noise = phas.new_image_like(np.angle(comp_noise)/phase_scale)
        print(TAG, 'Save magnitude noise', magn_diff_file)
        ants.image_write(magn_noise, magn_diff_file)
        print(TAG, 'Save phase noise', phas_diff_file)
        ants.image_write(phas_noise, phas_diff_file)





def main():
    parser = argparse.ArgumentParser(
    description="Performs complex-valued NLM denoising (using ants' denoise_image)"
                "of the input data specified by magnitude and phase. The output "
                "phase is going to be scaled in accordance with the input phase "
                "scaling (by default -4096...4096, but can be specified). "
                "Supports multiple contrasts and \"short\" and \"short\" averaging "
                "contrast measurements order. By default, a phase drift correction "
                "is performed prior to averaging, that can be switched off, however. "
                "Optionally, the magnitude and phase noise as the difference between the "
                "input data are denoised data can be stored.")
    parser.add_argument('magn_file', type=str)
    parser.add_argument('phas_file', type=str)
    parser.add_argument('--scale-phase', '-s', type=float, default=np.pi/4096.0,
                        help="Default: pi/4096.0")
    parser.add_argument('--out', '-o', type=str, default=None,
    help =      'Specify a common base name for all outputs instead of the input '
                'file names of magnitude and phase as the default basis.')
    parser.add_argument('--background-free', '-b', action='store_true',
    help =      'Consider the input phase already background-free')
    parser.add_argument('--diff', '-d', action='store_true',
    help =      'Save a magnitude and phase noise as the complex difference of input and output.')

    args = parser.parse_args()

    complex_antsnlm_denoise(args.magn_file, args.phas_file, args.scale_phase, args.out, args.diff, args.background_free)


if __name__ == '__main__':
    sys.exit(main())
