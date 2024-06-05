# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import nibabel as nib
import numpy as np
import argparse
import sys
from scipy.ndimage import gaussian_filter

TAG = '[CAVG]'

def complex_nifti_average(magn_file, phas_file, phase_scale, contrasts, averaging='long', out=None, diff=False, phase_corr=False, volumes=[0,-1]):

    print(TAG, 'Load magnitude')
    nii = nib.load(magn_file)
    magn = nii.get_fdata().astype(np.float32)
    print(TAG, 'Load phase')
    phas = nib.load(phas_file).get_fdata().astype(np.float32) * phase_scale

    comp = magn * np.exp(1.0j*phas)

    if out is None:
        magn_out_file = magn_file + '.mean.nii.gz'
        phas_out_file = phas_file + '.mean.nii.gz'
        magn_diff_file = magn_file + '.diff.nii.gz'
        phas_diff_file = phas_file + '.diff.nii.gz'
    else:
        magn_out_file = out + '.magn_mean.nii.gz'
        phas_out_file = out + '.phas_mean.nii.gz'
        magn_diff_file = out + 'magn_diff.nii.gz'
        phas_diff_file = out + 'phas_diff.nii.gz'

    if contrasts>0:
        new_sh = list(magn.shape[:-1]) + [contrasts] + [-1]
        print(TAG, 'Reshape to new shape:')
        if averaging == 'long':
            order = 'F'
        else:
            order = 'C'

        comp = np.reshape(comp, new_sh, order=order)
        print(TAG, comp.shape)

    new_sh = comp.shape
    comp[~np.isfinite(comp)]=0.0

    if volumes[1]==-1:
        volumes[1]=new_sh[-1]-1

    if diff or phase_corr:
        print(TAG, 'Magnitude and phase difference')
        magn_diff = np.diff(np.abs(comp[...,volumes[0]:volumes[1]+1]), axis=-1)
        phas_diff = np.angle(comp[...,volumes[0]+1:volumes[1]+1]*np.conj(comp[...,volumes[0]:volumes[1]]))

    if diff:
        print(TAG, 'Save difference magnitude', magn_diff_file)
        nib.save(nib.Nifti1Image(magn_diff, nii.affine, nii.header), magn_diff_file)

        print(TAG, 'Save difference phase', phas_diff_file)
        nib.save(nib.Nifti1Image(phas_diff/phase_scale, nii.affine, nii.header), phas_diff_file)

    if phase_corr:
        print(TAG, 'Phase matching')
        if isinstance(phase_corr, str):
            print(TAG, 'Load phase difference')
            phas_diff = nib.load(phase_corr).get_fdata().astype(np.float32) * phase_scale
            if np.ndim(phas_diff) < np.ndim(comp):
                phas_diff = phas_diff[...,np.newaxis]
        else:
            phas_diff = np.cumsum(phas_diff, axis=-1)

        sigma = np.zeros(len(phas_diff.shape))
        sigma[:3] = 2
        corr_real = gaussian_filter(np.cos(phas_diff), sigma=sigma)
        corr_imag = gaussian_filter(np.sin(phas_diff), sigma=sigma)
        corr = np.exp(1.0j*np.arctan2(corr_imag, corr_real))
        comp[...,volumes[0]+1:volumes[1]+1] *= np.conj(corr)

    print(TAG, 'Complex-valued average')
    comp = np.mean(comp[...,volumes[0]:volumes[1]+1], axis=-1)

    print(TAG, 'Save magnitude', magn_out_file)
    nib.save(nib.Nifti1Image(np.abs(comp), nii.affine, nii.header), magn_out_file)

    print(TAG, 'Save phase', phas_out_file)
    nib.save(nib.Nifti1Image(np.angle(comp)/phase_scale, nii.affine, nii.header), phas_out_file)


def main():
    parser = argparse.ArgumentParser(
    description="Performs the complex-valued average along the last dimension "
                "of the input data specified by magnitude and phase. The output "
                "phase is going to be scaled in accordance with the input phase "
                "scaling (by default -4096...4096, but can be specified). "
                "Supports multiple contrasts and \"short\" and \"short\" averaging "
                "contrast measurements order. By default, a phase drift correction "
                "is performed prior to averaging, that can be switched off, however. "
                "Optionally, the magnitude and phase differences between the "
                "input measurements are computed and stored.")
    parser.add_argument('magn_file', type=str)
    parser.add_argument('phas_file', type=str)
    parser.add_argument('--scale-phase', '-s', type=float, default=np.pi/4096.0)
    parser.add_argument('--volumes', '-v', nargs=2, type=int, default=[0, -1],
    help =      'The range of volumes to be averaged (first, last). Zero-index-based!')
    parser.add_argument('--contrasts', '-c', type=int, default=0,
    help =      'If the input contains multiple contrasts, e.g. multi-TE shots or MPM contrasts')
    parser.add_argument('--averaging', '-a', choices=['short', 'long'],
    help =      'Whether the same contrast repeats before the next contrast (short) '
                'or whether the contrasts permute before being repeated (long).')
    parser.add_argument('--out', '-o', type=str, default=None,
    help =      'Specify a common base name for all outputs instead of the input '
                'file names of magnitude and phase as the default basis.')
    parser.add_argument('--diff', '-d', action='store_true',
    help =      'Save a magnitude difference and phase difference file.')
    parser.add_argument('--phase-corr', '-p', nargs='?', type=str, default='True', const='True',
    help =      'By default, performs a phase drift correction to all but the first '
                'measurements prior to averaging. '
                'To this end, the phase difference from measurement to measurement '
                'is calculated, smoothed (2 voxel Gaussian sigma), and subtracted.'
                'Only, if you specify \'-p False\', you can disable this behaviour. '
                'You can also specify a phase difference file spcefically for '
                'this correction, i.e. if you want the same correction applied '
                'on different contrasts, e.g. the difference measured at one '
                'TE (stored with the -d option) applied to all other TEs. '
                'Note: this specifies the phase difference to be smoothed and subtracted, '
                'not a correction phase to be added.')

    args = parser.parse_args()

    if args.phase_corr.lower()=='false':
        phase_corr = False
    elif args.phase_corr.lower()=='true':
        phase_corr = True
    else:
        phase_corr = args.phase_corr


    complex_nifti_average(args.magn_file, args.phas_file, args.scale_phase, args.contrasts, args.averaging, args.out, args.diff, phase_corr, args.volumes)


if __name__ == '__main__':
    sys.exit(main())
