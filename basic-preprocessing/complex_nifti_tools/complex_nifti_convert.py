#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :   complex_nifti_convert.py
@Time    :   2023/06/28 17:23:39
@Author  :   Rüdiger Stirnberg 
@Contact :   ruediger.stirnberg@dzne.de
'''


import argparse
import sys
import numpy as np
import nibabel as nib


def complex_nifti_convert(magn_file, phas_file, factor=[1.0], complex_cartesian='none', apodization=False, phase_scale=None):
    from os import path
    import numpy.fft as ft
    import nibabel as nib
    import numpy as np
    def i2k(image, axes=[0,1,2]):
        k = ft.fftshift(ft.ifftn(ft.ifftshift(image, axes=axes), axes=axes), axes=axes)
        return k

    def k2i(kspace, axes=[0,1,2]):
        i = ft.fftshift(ft.fftn(ft.ifftshift(kspace, axes=axes), axes=axes), axes=axes)
        return i

    TAG = '[CCON]'
    if complex_cartesian == 'none' or complex_cartesian == 'out':
        print(TAG, 'Load magnitude')
        nii = nib.load(magn_file)
        magn = nii.get_fdata().astype(np.float32)
        if phas_file != None:
            print(TAG, 'Load phase')
            nii2 = nib.load(phas_file)
            phas = nii2.get_fdata().astype(np.float32)
            if phase_scale == None:
                # deduce phase scale from low magnitude phase voxels
                mask_background = magn<np.quantile(magn, 0.1)
                phase_background = phas[mask_background]
                # phase should be uniformly distributed [-pi, pi) with low magnitude SNR (e.g. air background)
                if np.max(np.abs(phase_background)) > 4:
                    print(TAG, 'Phase is not in radians. Assume pi=4096.0 dicom convention')
                    phase_scale = np.pi/4096.0
                else:
                    phase_scale = 1.0
            # apply phase scaling
            phas *= phase_scale
        else:
            phas = 0
            nii2 = nii

        comp = magn * np.exp(1.0j*phas)
    else:
        assert phas_file != None, "A real and imagrinary file must be specified."

        if phase_scale == None:
            phase_scale = 1.0

        print(TAG, 'Load real')
        nii = nib.load(magn_file)
        real = nii.get_fdata().astype(np.float32)
        print(TAG, 'Load imaginary')
        nii2 = nib.load(phas_file)
        imag = nii2.get_fdata().astype(np.float32)

        comp = real + 1.0j*imag

    dir1, name1 = path.split(magn_file)
    dir2, name2 = path.split(phas_file)
    if complex_cartesian == 'none' or complex_cartesian == 'both':
        out_file1 = path.abspath(path.join(dir1, 'ccon_' + name1))
        if phas_file != None:
            out_file2 = path.abspath(path.join(dir2, 'ccon_' + name2))
    elif complex_cartesian == 'in':
        out_file1 = path.abspath(path.join(dir1, 'ccon2magn_' + name1))
        out_file2 = path.abspath(path.join(dir1, 'ccon2phas_' + name1))
    elif complex_cartesian == 'out':
        out_file1 = path.abspath(path.join(dir1, 'ccon2real_' + name1))
        out_file2 = path.abspath(path.join(dir1, 'ccon2imag_' + name1))

    sh = np.array(comp.shape)
    outsh = sh.copy()
    factor = np.array(factor)
    outsh[:3] = np.round(np.array(sh[:3])/factor).astype(int)
    header = nii.header
    header2 = nii2.header

    if np.all(outsh == sh):
        print(TAG, 'No resampling required')
        img = comp
    else:
        print(TAG, 'Fourier transform with input shape:')
        print(TAG, sh)
        k = i2k(comp)
        K = np.zeros(outsh).astype(k.dtype)
        cropsh = outsh.copy()
        for j in range(3):
            cropsh[j] = np.min((outsh[j],sh[j]))
        k = k[sh[0]//2-cropsh[0]//2:sh[0]//2+cropsh[0]//2,
            sh[1]//2-cropsh[1]//2:sh[1]//2+cropsh[1]//2,
            sh[2]//2-cropsh[2]//2:sh[2]//2+cropsh[2]//2,...]
        
        if apodization:
            print(TAG, 'Hamming apodization along axis:')
            for j in range(3):
                if outsh[j] > sh[j]:
                    print(TAG, j)
                    x = np.linspace(-np.pi, np.pi, sh[j])
                    h = 0.46*np.cos(x)+0.54
                    k = np.moveaxis(k, source=j, destination=-1)
                    k *= h
                    k = np.moveaxis(k, source=-1, destination=j)


        K[outsh[0]//2-cropsh[0]//2:outsh[0]//2+cropsh[0]//2,
        outsh[1]//2-cropsh[1]//2:outsh[1]//2+cropsh[1]//2,
        outsh[2]//2-cropsh[2]//2:outsh[2]//2+cropsh[2]//2,...] = k
        outsh = K.shape

        # update factor according to actual rounded matrix sizes
        factor = sh[:3]/outsh[:3]

        print(TAG, 'Fourier transform back to output shape:')
        print(TAG, outsh)
        img = k2i(K)
        zooms = np.array(header.get_zooms())
        zooms[:3] *= factor
        sform = np.array(header.get_sform())
        shape = np.array(header.get_data_shape())
        for j in range(3):
            sform[j,j] *= factor[j]

        shape[:len(outsh)] = outsh

        for h in [header, header2]:
            h.set_zooms(zooms)
            h.set_sform(sform)
            h.set_data_shape(shape)
    
        print(TAG, 'Actual final voxel size:')
        print(TAG, zooms)

    if complex_cartesian == 'none' or complex_cartesian == 'in':
        if complex_cartesian == 'in':
            # output magnitude datatype
            header.set_datatype = 'uint16'
            # output phase datatype
            header2.set_datatype = 'float32'
        print(TAG, 'Save magnitude', out_file1)
        nib.save(nib.Nifti1Image(np.abs(img), None, header), out_file1)

        if phas_file != None:
            print(TAG, 'Save phase', out_file2)
            nib.save(nib.Nifti1Image(np.angle(img)/phase_scale, None, header2), out_file2)
    elif complex_cartesian == 'out' or complex_cartesian == 'both':
        if complex_cartesian == 'out':
            # output real and imag datatype
            header.set_datatype = 'float32'
            header2 = header
        print(TAG, 'Save real', out_file1)
        nib.save(nib.Nifti1Image(np.real(img), None, header), out_file1)

        print(TAG, 'Save imaginary', out_file2)
        nib.save(nib.Nifti1Image(np.imag(img), None, header2), out_file2)

    return out_file1, out_file2



def main():
    parser = argparse.ArgumentParser(
    description="Perform complex-valued conversion (between polar or cartesian reperesentation) with optional sinc-resampling (i.e. k-space cropping or zero padding).",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('magn_file', type=str)
    parser.add_argument('phas_file', type=str, nargs='?', default=None)
    group1 = parser.add_mutually_exclusive_group()
    group1.add_argument('--voxel-size', '-v', type=float, nargs='+', default=None, help='Output voxel size [mm] (1 for all dimensions = isotropic, or 3 for each dimension)')
    group1.add_argument('--resample-factor', '-d', type=float, nargs='+', default=[1], help='Resample scaling factor(s) of the output voxel size compared to the input voxel size (1 for all dimensions, or 3 for each dimension)')
    parser.add_argument('--apodization', '-a', action='store_true', help="Apply Hamming apodization (only before upsampling) to avoid Gibb's ringing")
    parser.add_argument('--scale-phase', '-s', type=float, nargs='?', default=None, const=np.pi/4096.0, help='(Input) Phase scale factor (output phase scale factor is reciprocal). By default (None), phase scale is automatically determined. Options: -s (without argument) = assume dicom pi=4096 convention; -s (with argument) = apply argument as tha scale factor')
    parser.add_argument('--complex-format', '-c', choices=['pol2cart', 'cart2pol', 'pol2pol', 'cart2cart'], help="'pol2cart: convert magn/phas to real/imag; 'cart2pol': convert real/imag to magn/phas; 'pol2pol': keep magn/phas; 'cart2cart: keep real/imag", default='pol2cart')

    args = parser.parse_args()


    out_size = args.voxel_size
    if out_size != None:
        assert len(out_size)==1 or len(out_size)==3, f"Either 1 (for isotropic) or 3 output voxel sizes need to be specified, got: {len(out_size)}"
        if len(out_size)==1:
            out_size = [out_size[0] for i in range(3)]

        nii = nib.load(args.magn_file)
        orig_size = nii.header.get_zooms()[:3]
        del nii
        rsf = np.array(out_size)/np.array(orig_size)

    else:
        rsf = args.resample_factor
        assert len(rsf)==1 or len(rsf)==3, f"Either 1 (for isotropic) or 3 resample factors need to be specified, got: {len(rsf)}"
    
        if len(rsf)==1:
            rsf = [rsf[0] for i in range(3)]

    apodization = args.apodization
    if np.all(np.array(rsf)>=1):
        apodization = False

    cform = args.complex_format.lower()
    if cform=='pol2pol':
        ccart = 'none'
    elif cform=='cart2cart':
        ccart = 'both'
    elif cform=='cart2pol':
        ccart = 'in'
    else:
        ccart = 'out'

    complex_nifti_convert(args.magn_file, args.phas_file, rsf, ccart, apodization, phase_scale=args.scale_phase)


if __name__ == '__main__':
    sys.exit(main())
