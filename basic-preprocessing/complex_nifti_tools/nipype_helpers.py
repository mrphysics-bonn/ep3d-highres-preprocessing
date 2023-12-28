#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :   nipype_helpers.py
@Time    :   2023/06/23 12:54:44
@Author  :   Rüdiger Stirnberg 
@Contact :   ruediger.stirnberg@dzne.de
'''

from curses.panel import new_panel
import os
import nipype.interfaces.fsl as fsl
import nipype.interfaces.utility as util     # utility
import nipype.pipeline.engine as pe          # pypeline engine


# WORKFLOWS

def convert_complex_polar(base_dir=os.getcwd(), name="convert_complex_polar"):
    convert_complex_polar = pe.Workflow(name=name)
    convert_complex_polar.base_dir = base_dir

    # Set up a node to define all inputs required for this workflow
    inputnode = pe.Node(interface=util.IdentityInterface(fields=['magn_file', 'phas_file']), name='inputnode')

    # Magn/Phas to Real/Imag
    pola2cplx = pe.Node(interface=fsl.utils.Complex(), name='pola2cplx')
    pola2cplx.inputs.complex_polar = True
    cplx2cart = pe.Node(interface=fsl.utils.Complex(), name='cplx2cart')
    cplx2cart.inputs.real_cartesian = True

    convert_complex_polar.connect(inputnode   , 'magn_file'           , pola2cplx , 'magnitude_in_file'   )
    convert_complex_polar.connect(inputnode   , 'phas_file'           , pola2cplx , 'phase_in_file'       )
    convert_complex_polar.connect(pola2cplx   , 'complex_out_file'    , cplx2cart , 'complex_in_file'     )

    # Copy original affine back to Real/Imag
    cpgeom_real = pe.Node(interface=fsl.utils.CopyGeom(), name='cpgeom_real')
    cpgeom_imag = pe.Node(interface=fsl.utils.CopyGeom(), name='cpgeom_imag')
    convert_complex_polar.connect(cplx2cart   , 'real_out_file'       , cpgeom_real, 'dest_file' )
    convert_complex_polar.connect(inputnode   , 'magn_file'           , cpgeom_real, 'in_file'   )
    convert_complex_polar.connect(cplx2cart   , 'imaginary_out_file'  , cpgeom_imag, 'dest_file' )
    convert_complex_polar.connect(inputnode   , 'magn_file'           , cpgeom_imag, 'in_file'   )

    # Set up a node to define all outputs of this workflow
    outputnode = pe.Node(interface=util.IdentityInterface(fields=['real_file', 'imag_file']), name="outputnode")
    convert_complex_polar.connect(cpgeom_real , 'out_file'            , outputnode  , 'real_file' )
    convert_complex_polar.connect(cpgeom_imag , 'out_file'            , outputnode  , 'imag_file' )

    return convert_complex_polar

def convert_complex_cartesian(base_dir=os.getcwd(), name="convert_complex_cartesian"):
    convert_complex_cartesian = pe.Workflow(name=name)
    convert_complex_cartesian.base_dir = base_dir

    # Set up a node to define all inputs required for this workflow
    inputnode = pe.Node(interface=util.IdentityInterface(fields=['real_file', 'imag_file']), name='inputnode')

    # Real/Imag to Magn/Phas
    cart2cplx = pe.Node(interface=fsl.utils.Complex(), name='cart2cplx')
    cart2cplx.inputs.complex_cartesian = True
    cplx2pola = pe.Node(interface=fsl.utils.Complex(), name='cplx2pola')
    cplx2pola.inputs.real_polar = True

    convert_complex_cartesian.connect(inputnode   , 'real_file'           , cart2cplx , 'real_in_file'   )
    convert_complex_cartesian.connect(inputnode   , 'imag_file'           , cart2cplx , 'imaginary_in_file')
    convert_complex_cartesian.connect(cart2cplx   , 'complex_out_file'    , cplx2pola , 'complex_in_file')

    # Copy original affine back to Magn/Phas
    cpgeom_magn = pe.Node(interface=fsl.utils.CopyGeom(), name='cpgeom_magn')
    cpgeom_phas = pe.Node(interface=fsl.utils.CopyGeom(), name='cpgeom_phas')
    convert_complex_cartesian.connect(cplx2pola   , 'magnitude_out_file'  , cpgeom_magn, 'dest_file' )
    convert_complex_cartesian.connect(inputnode   , 'real_file'           , cpgeom_magn, 'in_file'   )
    convert_complex_cartesian.connect(cplx2pola   , 'phase_out_file'      , cpgeom_phas, 'dest_file' )
    convert_complex_cartesian.connect(inputnode   , 'real_file'           , cpgeom_phas, 'in_file'   )

    # Set up a node to define all outputs of this workflow
    outputnode = pe.Node(interface=util.IdentityInterface(fields=['magn_file', 'phas_file']), name="outputnode")
    convert_complex_cartesian.connect(cpgeom_magn , 'out_file'            , outputnode  , 'magn_file' )
    convert_complex_cartesian.connect(cpgeom_phas , 'out_file'            , outputnode  , 'phas_file' )

    return convert_complex_cartesian

# FUNCTIONS

def phasematch_cartesian_function(real_file, imag_file, ref_vol=0):
    import nibabel as nib
    import numpy as np
    from scipy.ndimage import gaussian_filter
    import os

    nii = nib.load(real_file)
    real = nii.get_fdata().astype(np.float32)
    imag = nib.load(imag_file).get_fdata().astype(np.float32)
    comp = real + 1.0j * imag

    if ref_vol!=0:
        comp = np.roll(comp, shift=-ref_vol, axis=-1)

    phas_diff = np.angle(comp[...,1:]*np.conj(comp[...,:-1]))

    sigma = np.zeros(len(phas_diff.shape))
    sigma[:3] = 2
    corr_real = gaussian_filter(np.cos(phas_diff), sigma=sigma)
    corr_imag = gaussian_filter(np.sin(phas_diff), sigma=sigma)
    comp[...,1:] *= np.exp(-1.0j*np.arctan2(corr_imag, corr_real))

    if ref_vol!=0:
        comp = np.roll(comp, shift=ref_vol, axis=-1)

    real_out_file = os.path.join(os.getcwd(),'phasematched_'+os.path.split(real_file)[1])
    imag_out_file = os.path.join(os.getcwd(),'phasematched_'+os.path.split(imag_file)[1])

    nib.save(nib.Nifti1Image(np.real(comp) , affine=nii.affine, header=nii.header), real_out_file)
    nib.save(nib.Nifti1Image(np.imag(comp) , affine=nii.affine, header=nii.header), imag_out_file)

    return real_out_file, imag_out_file

def mean_cartesian_function(real_file, imag_file):
    import nibabel as nib
    import numpy as np
    import os

    nii = nib.load(real_file)
    real = nii.get_fdata().astype(np.float32)
    imag = nib.load(imag_file).get_fdata().astype(np.float32)
    
    real_out = np.mean(real, axis=-1)
    imag_out = np.mean(imag, axis=-1)

    real_out_file = os.path.join(os.getcwd(),'mean_'+os.path.split(real_file)[1])
    imag_out_file = os.path.join(os.getcwd(),'mean_'+os.path.split(imag_file)[1])

    nib.save(nib.Nifti1Image(real_out, affine=nii.affine, header=nii.header), real_out_file)
    nib.save(nib.Nifti1Image(imag_out, affine=nii.affine, header=nii.header), imag_out_file)

    return real_out_file, imag_out_file

def denoise_cartesian_function(real_file, imag_file):
    import numpy as np
    import os
    import ants
    from skimage.restoration import unwrap_phase
    from scipy.ndimage import gaussian_filter
    TAG = '[NLM(nipype)]'

    real_out_file = os.path.join(os.getcwd(),'denoised_'+os.path.split(real_file)[1])
    imag_out_file = os.path.join(os.getcwd(),'denoised_'+os.path.split(imag_file)[1])

    print(TAG, 'Load real and imag')
    real_img = ants.image_read(real_file)
    imag_img = ants.image_read(imag_file)

    comp = real_img.numpy() + 1.0j * imag_img.numpy()
    magn = real_img.new_image_like(np.abs(comp))

    print(TAG, 'Compute unwrapped and smoothed background phase approximation')
    phas_baseline = unwrap_phase(np.angle(comp))
    phas_baseline = gaussian_filter(phas_baseline,4)

    print(TAG, 'Temporarily compute real and imaginary images w/o background phase')
    real = np.real(comp * np.exp(-1.0j*phas_baseline))/2.0+4096/2
    imag = np.imag(comp * np.exp(-1.0j*phas_baseline))/2.0+4096/2
    real = real_img.new_image_like(real)
    imag = imag_img.new_image_like(imag)
    del comp

    print(TAG, 'Get a mask from the manitude image')
    mask = ants.get_mask(magn, low_thresh=magn.mean()/5.0) #include even dark regions, but not all voxels outside the brain (it just takes too long to process)

    print(TAG, "Denoise real and imag image")
    real_denoise = ants.denoise_image(real, mask, noise_model='Gaussian')
    real_denoise = real_denoise*2-4096
    imag_denoise = ants.denoise_image(imag, mask, noise_model='Gaussian')
    imag_denoise = imag_denoise*2-4096
    del mask, real, imag

    print(TAG, "Add background phase back")
    comp_denoise = (real_denoise.numpy()) + 1.0j*(imag_denoise.numpy())
    comp_denoise *= np.exp(1.0j*phas_baseline)
    del phas_baseline

    real_denoise = real_img.new_image_like(np.real(comp_denoise))
    imag_denoise = imag_img.new_image_like(np.imag(comp_denoise))
    del comp_denoise, real_img, imag_img

    
    print(TAG, 'Save real and imag image')
    ants.image_write(real_denoise, real_out_file)
    ants.image_write(imag_denoise, imag_out_file)

    return real_out_file, imag_out_file


def extension(filename):
    from pathlib import Path
    ext = "".join([s for s in Path(filename).suffixes if s in ['.nii', '.gz']])
    return ext

# erode mpr_mask in QSM space
def binary_erosion(mask_file, iterations=1):
    import nibabel as nib
    from scipy.ndimage import binary_erosion
    from complex_nifti_tools.nipype_helpers import extension
    import os
    
    nii = nib.load(mask_file)
    msk = nii.get_fdata() > 0
    msk = binary_erosion(msk, iterations=iterations)
    
    ext = extension(mask_file)
    out_file = mask_file.split(ext)[0]
    out_file = os.path.join(os.getcwd(),os.path.split(out_file)[1])
    out_file += '_erode' + ext
    nib.save(nib.Nifti1Image(msk, nii.affine, nii.header), out_file)

    return out_file
