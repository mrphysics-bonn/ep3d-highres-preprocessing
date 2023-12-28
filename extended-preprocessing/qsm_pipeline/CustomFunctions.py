# -*- coding: utf-8 -*-
"""
Created on Mon Dec 3 12:27:00 2018

@authors: stirnbergr, shahidm
@email: ruediger.stirnberg@dzne.de
"""
from __future__ import print_function
import numpy as np
import numpy.fft as ft
import nibabel as nib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import os

def hip(euler1, euler2):
    hip = euler1 * np.conj(euler2)
    return hip

def simple_complex_phasediff(P1, P2, offset=0.0):
    dP = P1-P2-offset

    ret = dP + offset

    case = dP > np.pi
    ret[case] -= 2.0*np.pi

    case = dP < -np.pi
    ret[case] += 2.0*np.pi

    return np.remainder(ret + np.pi, 2.0*np.pi) - np.pi

def hip_phase(phase1, phase2):
    return simple_complex_phasediff(phase1, phase2)

def get_comp_ro_kspace(pc_AP_magn_file, pc_PA_magn_file, pc_AP_phas_file, pc_PA_phas_file):

    magn = nib.load(pc_AP_magn_file)
    phas = nib.load(pc_AP_phas_file)
    comp_ro = magn.get_fdata(caching='unchanged', dtype=np.float32) * np.exp(1.0j * phas.get_fdata(caching='unchanged', dtype=np.float32))

    magn = nib.load(pc_PA_magn_file)
    phas = nib.load(pc_PA_phas_file)
    #print(comp_ro.shape)
    #print(magn.shape)
    tmp = magn.get_fdata(caching='unchanged', dtype=np.float32) * np.exp(1.0j * phas.get_fdata(caching='unchanged', dtype=np.float32))
    affine = magn.affine

    del magn, phas

    comp_ro = np.append(comp_ro[...,np.newaxis], tmp[...,np.newaxis], axis=-1)
    #print(comp_ro.shape)


    comp_ro = np.mean(comp_ro, axis=-2)
    #print(comp_ro.shape)
    return affine, comp_ro


def get_ka_kd_kspace(comp_ro):

    k_a = ft.fftshift(ft.fftn(comp_ro[...,0], axes=[0,1,2]), axes=[0,1,2])
    k_d = ft.fftshift(ft.fftn(comp_ro[...,1], axes=[0,1,2]), axes=[0,1,2])
    return k_a, k_d

def get_mean_ro_pe_kspace(shape,comp_ro):

    k_a,k_d = get_ka_kd_kspace(comp_ro)
    RPE=3
    bPFfilter = False

    if bPFfilter:
        k_a[:,:2*shape[1]//8+RPE,...] = 0.0
        k_d[:,6*shape[1]//8-RPE:,...] = 0.0

    k = k_a + k_d
    mean_ro_pe = ft.ifftn(ft.fftshift(k/2.0, axes=[0,1,2]), axes=[0,1,2])
    #print(shape)
    #print(k.shape)
    return k_a, k_d, mean_ro_pe


def kspace_combine(pc_AP_magn_file, pc_PA_magn_file, pc_AP_phas_file, pc_PA_phas_file, header):

    import nibabel as nib
    import numpy as np
    import numpy.fft as ft
    from qsm_pipeline.CustomFunctions import get_comp_ro_kspace,get_mean_ro_pe_kspace
    import os

    #output_files=[]

    affine, comp_ro = get_comp_ro_kspace(pc_AP_magn_file, pc_PA_magn_file,
                                         pc_AP_phas_file, pc_PA_phas_file)


    shape = comp_ro.shape

    # unfiltered mean image
    #mean_comp = np.mean(comp_ro, axis=[-2,-1])
    #mean_magn_file = os.path.join(os.getcwd(),'mean_magn.nii.gz')
    #mean_phas_file = os.path.join(os.getcwd(),'mean_phas.nii.gz')
    #nib.save(nib.Nifti1Image(np.abs(mean_comp), affine = affine), path.join(sub_dir, 'interim', mean_magn_file))
    #nib.save(nib.Nifti1Image(np.angle(mean_comp), affine = affine), path.join(sub_dir, 'interim',mean_phas_file))

    #output_files.append(mean_magn_file)
    #output_files.append(mean_phas_file)

    # print(np.ndim(comp_ro))
    #phas_diff = 0
    bPFfilter = False

    k_a, k_d, mean_ro_pe = get_mean_ro_pe_kspace(shape,comp_ro)
    #k = k_a + k_d


    A = np.abs(k_a)
    D = np.abs(k_d)
    print(A.shape)
    A = np.sqrt(np.mean(A**2, axis=(0,2,3)))
    D = np.sqrt(np.mean(D**2, axis=(0,2,3)))

    P = 1
    f_a = A**P/(A**P+D**P)

    f_d = D**P/(A**P+D**P)

    del A, D


    k_filt  = np.rollaxis(k_a, axis=1, start=k_a.ndim) * f_a
    k_filt += np.rollaxis(k_d, axis=1, start=k_d.ndim) * f_d
    k_filt  = np.rollaxis(k_filt, axis=-1, start=1)

    mean_ro_pe_filt = ft.ifftn(ft.fftshift(k_filt, axes=[0,1,2]), axes=[0,1,2])

    mean_magn_filt_file = os.path.join(os.getcwd(),'mean_magn_filt%d.nii.gz'%(int(bPFfilter)))
    mean_phas_filt_file = os.path.join(os.getcwd(),'mean_phas_filt%d.nii.gz'%(int(bPFfilter)))
    mean_ro_pe_magn_file = os.path.join(os.getcwd(),'mean_ro_pe_magn.nii.gz')

    nib.save(nib.Nifti1Image(np.abs(mean_ro_pe_filt), affine = affine, header=header), mean_magn_filt_file)
    nib.save(nib.Nifti1Image(np.angle(mean_ro_pe_filt), affine = affine, header=header), mean_phas_filt_file)
    nib.save(nib.Nifti1Image(np.abs(mean_ro_pe), affine = affine, header=header), mean_ro_pe_magn_file)


    """
    rmse_magn_filt = np.sqrt(np.mean(np.abs(mean_ro_pe_filt)**2, axis=-1))

    rms_magn_filt_file = os.path.join(os.getcwd(),'rms_magn_filt%d.nii.gz'%(int(bPFfilter)))
    nib.save(nib.Nifti1Image(rmse_magn_filt, affine = affine), rms_magn_filt_file)

    output_files.append(rms_magn_filt_file)

    rmse_magn = np.sqrt(np.mean(np.abs(mean_ro_pe)**2, axis=-1))
    rms_magn_file = os.path.join(os.getcwd(),'rms_magn%d.nii.gz'%(int(bPFfilter)))
    nib.save(nib.Nifti1Image(rmse_magn, affine = affine), rms_magn_file)

    output_files.append(rms_magn_file)

    bPFfilter = False

    mean_ro_pe_filt = nib.load(mean_magn_filt_file)

    affine = mean_ro_pe_filt.affine
    mean_ro_pe_filt = mean_ro_pe_filt.get_fdata() * np.exp(1.0j * nib.load(mean_phas_filt_file).get_fdata())
    pdiff_filt = np.angle(np.mean(hip(mean_ro_pe_filt[...,1:], mean_ro_pe_filt[...,:-1]), axis=-1))
    pdiff_phas_filt_file = os.path.join(os.getcwd(),'pdiff_phas_filt%d.nii.gz'%(int(bPFfilter)))
    nib.save(nib.Nifti1Image(pdiff_filt, affine = affine), pdiff_phas_filt_file)
    output_files.append(pdiff_phas_filt_file)


    gre_magn = nib.load(path.join(sub_dir, 'raw', '033-QSM.nii.gz'))
    gre_affine = gre_magn.affine
    gre_rms = np.sqrt(np.mean(gre_magn.get_fdata().astype(float)**2, axis=-1))
    nib.save(nib.Nifti1Image(gre_rms, gre_affine),path.join(sub_dir, 'interim', 'gre_rms.nii.gz'))

    raw_magn = nib.load(path.join(sub_dir, 'interim', 'mean_magn_raw.nii.gz'))
    raw_affine = raw_magn.affine
    raw_rms = np.sqrt(np.mean(raw_magn.get_fdata().astype(float)**2, axis=-1))
    nib.save(nib.Nifti1Image(raw_rms, raw_affine), path.join(sub_dir, 'interim', 'raw_rms.nii.gz'))
    """

    return mean_magn_filt_file, mean_phas_filt_file, mean_ro_pe_magn_file

def contrast_combine(magn_file, phas_file, header, multi_echo_shots=1):
    import nibabel as nib
    import numpy as np
    import os
    magn = nib.load(magn_file)
    phas = nib.load(phas_file)
    affine = magn.affine
    comp = magn.get_fdata(caching='unchanged', dtype=np.float32) * np.exp(1.0j * phas.get_fdata(caching='unchanged', dtype=np.float32))
    del magn, phas

    sh = np.array(comp.shape)

    multi_echoes = sh[3]//multi_echo_shots
    sh[3] = multi_echoes
    comb = np.zeros(sh, dtype=np.complex64)
    sh[3] = multi_echo_shots
    rmste = np.zeros(sh, dtype=np.float32)
    for tseg in range(multi_echo_shots):
        comb += comp[:,:,:,tseg::multi_echo_shots]
        rmste[...,tseg] = np.sqrt(np.mean(np.abs(comp[:,:,:,tseg::multi_echo_shots])**2, axis=-1))
    comb /= multi_echo_shots
    rms = np.sqrt(np.mean(rmste**2, axis=-1))


    comb_magn_file = os.path.join(os.getcwd(), 'comb_magn.nii.gz')
    comb_phas_file = os.path.join(os.getcwd(), 'comb_phas.nii.gz')
    rmste_file = os.path.join(os.getcwd(), 'rmste.nii.gz')
    rms_file   = os.path.join(os.getcwd(), 'rms.nii.gz')


    print("Save combined files", rmste.shape)
    nib.save(nib.Nifti1Image(np.abs(comb), affine, header=header), comb_magn_file)
    nib.save(nib.Nifti1Image(np.angle(comb), affine, header=header), comb_phas_file)
    nib.save(nib.Nifti1Image(rmste, affine, header=header), rmste_file)
    nib.save(nib.Nifti1Image(rms, affine, header=header), rms_file)

    return comb_magn_file, comb_phas_file, rmste_file, rms_file

def complex_average(magn_file, phas_file, header, axis=-1):
    import nibabel as nib
    import numpy as np
    import os

    magn = nib.load(magn_file)
    phas = nib.load(phas_file)
    affine = magn.affine
    comp = magn.get_fdata(caching='unchanged', dtype=np.float32) * np.exp(1.0j * phas.get_fdata(caching='unchanged', dtype=np.float32))
    del magn, phas

    comp = np.mean(comp, axis=axis)

    mean_magn_file = os.path.join(os.getcwd(),'mean_magn.nii.gz')
    mean_phas_file = os.path.join(os.getcwd(),'mean_phas.nii.gz')

    nib.save(nib.Nifti1Image(np.abs(comp), affine = affine, header=header), mean_magn_file)
    nib.save(nib.Nifti1Image(np.angle(comp), affine = affine, header=header), mean_phas_file)

    auxiliary_output_files = []

    return mean_magn_file, mean_phas_file, auxiliary_output_files


def get_comp_phase_correction(applytopup_corrected_files, multi_echoes):

    applytopup_real_ap = ''
    applytopup_real_pa = ''
    applytopup_imag_ap = ''
    applytopup_imag_pa = ''

    for ac in applytopup_corrected_files:
        if 'real_mc_AP' in ac: applytopup_real_ap = ac
        if 'real_mc_PA' in ac: applytopup_real_pa = ac
        if 'imag_mc_AP' in ac: applytopup_imag_ap = ac
        if 'imag_mc_PA' in ac: applytopup_imag_pa = ac

    applytopup_real = nib.load(applytopup_real_ap)
    affine = applytopup_real.affine
    shape = np.array(applytopup_real.shape)
    shape[3] = multi_echoes
    shape = np.append(shape, np.array([-1]), axis=0)

    applytopup_real = np.reshape(applytopup_real.get_fdata(caching='unchanged', dtype=np.float32), shape, order='F')
    applytopup_real_pa = np.reshape(nib.load(applytopup_real_pa).get_fdata(caching='unchanged', dtype=np.float32), shape, order='F')
    if applytopup_real.ndim<5: # TE dimension is last (no separate measurements per PE direction). Create new axis
        applytopup_real = np.expand_dims(applytopup_real, axis=4)
        applytopup_real_pa = np.expand_dims(applytopup_real_pa, axis=4)
    applytopup_real = np.append(applytopup_real, applytopup_real_pa, axis=-1)
    print(applytopup_real.shape)

    applytopup_imag = np.reshape(nib.load(applytopup_imag_ap).get_fdata(caching='unchanged', dtype=np.float32), shape, order='F')
    applytopup_imag_pa = np.reshape(nib.load(applytopup_imag_pa).get_fdata(caching='unchanged', dtype=np.float32), shape, order='F')
    if applytopup_imag.ndim<5: # TE dimension is last (no separate measurements per PE direction). Create new axis
        applytopup_imag = np.expand_dims(applytopup_imag, axis=4)
        applytopup_imag_pa = np.expand_dims(applytopup_imag_pa, axis=4)
    applytopup_imag = np.append(applytopup_imag, applytopup_imag_pa, axis=-1)

    comp = applytopup_real + 1.0j * applytopup_imag
    print(comp.shape)
    return affine, comp

def get_comp_corr_frequency_correcion(comp0, TE=None):
    from scipy.ndimage.filters import gaussian_filter
    print("enter get_comp_corr_frequency_correcion")
    # 1. Determine frequency (times Delta TE) per self-contained measurment (t-seg and repetition).
    #    This is a wrapped Delta Phase image (with anatomical structure)
    # 0      | f1 (t2-t1)    | f1 (t3-t2)      | f1 (t4-t3)
    # 0      | f2 (t2-t1)    | f2 (t3-t2)      | f2 (t4-t3)
    # 0      | f3 (t2-t1)    | f3 (t3-t2)      | f3 (t4-t3)
    # ...
    HiP_dPhase_dTE = np.zeros_like(comp0)
    HiP_dPhase_dTE[...,1:,:] = comp0[...,1:,:]*np.conj(comp0[...,:-1,:])

    # ASSUMING that rephased multi-TE Delta TE is constant, average across TEs and
    # save wrapped frequency (times TE) per time point
    freq_times_DeltaTE = np.angle(np.mean(HiP_dPhase_dTE, axis=-2))

    # 2. Determine frequency changes (times Delta TE) with respect to the previous time point per measurement.
    #    These should be wrap-free maps (without anatomical structure)
    # 0      | 0             | 0               | 0
    # 0      | (f2-f1)(t2-t1)| (f2-f1)(t3-t2)  | (f2-f1)(t4-t3)
    # 0      | (f3-f2)(t2-t1)| (f3-f2)(t3-t2)  | (f3-f2)(t4-t3)
    # ...
    dPhase_dTE_dt = np.zeros(comp0.shape)
    HiP_dPhase_dTE_dt = np.zeros_like(comp0)
    HiP_dPhase_dTE_dt[...,1:] = HiP_dPhase_dTE[...,1:]*np.conj(HiP_dPhase_dTE[...,:-1])
    dPhase_dTE_dt[...,1:,1:] = np.angle(HiP_dPhase_dTE_dt[...,1:, 1:])

    weights = np.abs(HiP_dPhase_dTE_dt)
    del HiP_dPhase_dTE_dt

    print("some arrays defined")
    #
    # 3. Unwrap the frequency change maps across time points (1D)
    dPhase_dTE_dt = np.unwrap(dPhase_dTE_dt, axis=-1)
    print("unwrapped")
    #

    # If no TEs are specified, we have to live with the fact that we can't determine useful integration
    # constants at this stage. Thus leave the first column (t1) zero for all timepoints (2...).
    # Furthermore, we have to assume that Delta TE may vary from columns to collumn. Hence, we
    # cannot use a mean frequency per timepoint, but leave an individual frequency change (times Delta TE)
    # per column and row.
    # In a following step in the calling function get_comp_corr_phase_correction, the phases of
    # measurements that correspond to each other are matched (which overwrites the corresponding integration
    # constants). However, the downsides are that this subsequent step involves another smoothing stage,
    # and that a potential offset will remain between the first measurements of unique contrasts/tsegs.
    #
    # However, if TEs are defined, we can calculate the actual mean frequency change per time point
    # and we can enter useful integration constants in the first column (will become correct phase offsets
    # per timepoint after integration)
    if TE is not None:
        print("deal with actual TEs (part 1)")
        dTE = np.zeros_like(TE)
        dTE[1:,:] = np.diff(TE, axis=-2)
        print(TE.shape)
        print(TE, dTE)

        norm = np.sum(weights, axis=-2)

        print(norm.shape, weights.shape, dPhase_dTE_dt.shape)

        for m in range(1,dPhase_dTE_dt.shape[-1]): # time points
            for t in range(1,dPhase_dTE_dt.shape[-2]): # echo times
                weights[...,t,m] /= norm[...,m]
                weights[~np.isfinite(weight[...,t,m]),t,m] = 0.0 # avoid nans or infinites in dPhase_dTE_dt below
                dPhase_dTE_dt[...,t,m] /= (dTE[...,t,m]*2.0*np.pi) # actual frequency change now
            # use first zero column to store the mean across TEs
            dPhase_dTE_dt[...,0,m] = np.sum(weights[...,1:,m]*dPhase_dTE_dt[...,1:,m], axis=-1)
    else:
        del weights


    # 4. Smooth the frequency change maps across voxels
    gauss_sigma = np.array([2, 2, 2, 0, 0])
    dPhase_dTE_dt = gaussian_filter(dPhase_dTE_dt, gauss_sigma)
    print("filter applied")

    newshape = np.array(dPhase_dTE_dt.shape)[:-1]
    newshape[-1] = -1
    freq_change_times_DeltaTE = np.reshape(dPhase_dTE_dt, newshape, order='F')
    print("reshaped")

    #
    # 5. Integrate along time points (cumsum)
    # 0      | 0             | 0               | 0
    # 0      |s(f2-f1)(t2-t1)|s(f2-f1)(t3-t2)  |s(f2-f1)(t4-t3)
    # 0      |s(f3-f1)(t2-t1)|s(f3-f1)(t3-t2)  |s(f3-f1)(t4-t3)
    # ...
    dPhase_dTE = np.cumsum(dPhase_dTE_dt, axis=-1)
    print("cumsum1")

    if TE is not None:
        print("deal with actual TEs (part 2)")
        # construct integrated phase array manually based on TEs and mean frequency change
        # (with respect to first measurement)
        phase = np.zeros_like(dPhase_dTE)
        for m in range(1,dPhase_dTE_dt.shape[-1]): # time points
            for t in range(0,dPhase_dTE_dt.shape[-2]): # echo times
                phase[...,t,m] = dPhase_dTE[...,0,m] * 2*np.pi * TE[t,m]

        # Alternative version where individual phase differences are kept per TEintegration
        # and integration still has to be performed in the nexg step (6.)
        #for m in range(1,dPhase_dTE_dt.shape[-1]): # time points
        #    dPhase_dTE[...,0,m] *= 2*np.pi * TE[0,m]
        #    for t in range(1,dPhase_dTE_dt.shape[-2]): # echo times
        #        dPhase_dTE[...,t,m] *= 2*np.pi * dTE[t,m]
    else:
        # 6. Integrate along echo times (cumsum)
        # 0      | 0             | 0               | 0
        # 0      |s(f2-f1)(t2-t1)|s(f2-f1)(t3-t1)  |s(f2-f1)(t4-t1)
        # 0      |s(f3-f1)(t2-t1)|s(f3-f1)(t3-t1)  |s(f3-f1)(t4-t1)
        # ...
        phase = np.cumsum(dPhase_dTE, axis=-2)
        print("cumsum2")
    del dPhase_dTE, dPhase_dTE_dt

    #
    # 7. Subtract from original phase
    # f1 t1 | f1 t2            | f1 t3            | f1 t4
    # f2 t1 | f2 t2            | f2 t3            | f2 t4
    #       |-s(f2 t2)         |-s(f2 t3)         |-s(f2 t4)         # this cancels with original phase except for original noise
    #       |+s(f1 t2)         |+s(f1 t3)         |+s(f1 t4)         # this is the first time point without noise
    #       |+s(f2 t1 - f1 t1) |+s(f2 t1 - f1 t1) |+s(f2 t1 - f1 t1) # this is the 2nd time point phase offset of TE1 compared to the 1st time point without noise
    # f3 t1 | f3 t2            | f3 t3            | f3 t4
    #       |-s(f3 t2)         |-s(f3 t3)         |-s(f2 t4)         # this cancels with original phase except for original noise
    #       |+s(f1 t2)         |+s(f1 t3)         |+s(f1 t4)         # this is the first time point without noise
    #       |+s(f3 t1 - f1 t1) |+s(f3 t1 - f1 t1) |+s(f3 t1 - f1 t1) # this is the 3rd time point phase offset of TE1 compared to the 1st time point without noise
    # ...

    print(comp0.shape)
    print(phase.shape)
    comp0 *= np.exp(-1.0j * phase)
    print("times phase")

    HiP_dPhase_dTE[...,1:,:] = comp0[...,1:,:]*np.conj(comp0[...,:-1,:])
    print("more arrays defined")

    # ASSUMING that rephased multi-TE Delta TE is constant, average across TEs and
    # save unwrapped frequency (times TE) correction per time point
    freq_times_DeltaTE_corr = np.angle(np.mean(HiP_dPhase_dTE, axis=-2))

    return comp0, freq_times_DeltaTE, freq_times_DeltaTE_corr, freq_change_times_DeltaTE

def get_comp_corr_phase_correction(comp, multi_echo_shots=1, echo_times=None):
    from scipy.ndimage.filters import gaussian_filter

    print(comp.shape)

    rms = np.sqrt(np.mean(np.abs(comp)**2, axis=-1)) * comp.shape[-1]

    Q_before = np.abs(np.sum(comp, axis=-1)) / rms
    #print(comp_roll.shape)

    # NEW!!!
    # 0. Reshape original phase
    # f1 t1 | f1 t2          | f1 t3           | f1 t4 # ti: rephased TEs
    # f2 t1 | f2 t2          | f2 t3           | f2 t4
    # f3 t1 | f3 t2          | f3 t3           | f3 t4
    # fj: self0contained time points in chronological order,
    # i.e tsegs and meas. repetitions (with or without polarity changes)
    # ...
    sh=np.array(comp.shape)
    sh[3] = sh[3]//multi_echo_shots
    sh[4] = sh[4]*multi_echo_shots
    comp0 = np.zeros(sh, dtype=np.complex64)
    print(comp0.shape)

    TE = np.zeros(sh[3:], dtype=float)
    rephased_echoes = sh[3]
    for meas in range(comp.shape[4]):
        for tseg in range(multi_echo_shots):
            comp0[:,:,:,:,meas*multi_echo_shots+tseg] = comp[:,:,:,tseg::multi_echo_shots,meas]

            #constuct echo time array that corresponds to the comp0 array (just without spatial dimensions)
            if echo_times is not None:
                TE[:, meas*multi_echo_shots+tseg] = echo_times[:rephased_echoes]
                if tseg>0:
                    TE[:, meas*multi_echo_shots+tseg] += (echo_times[rephased_echoes+tseg-1] - echo_times[0])

        print("get new dimension of comp0 %i" % meas)

    if echo_times is not None:
        TE *= 1.0e-3 #ms -> s
        print("constructed actual echo time matrix to be passed to get_comp_corr_frequency_correcion:")
        print(TE)
    else:
        TE = None

    comp0, freq_times_DeltaTE, freq_times_DeltaTE_corr, freq_change_times_DeltaTE = get_comp_corr_frequency_correcion(comp0, TE)
    print("got get_comp_corr_frequency_correcion")

    # Reshape back
    for meas in range(comp.shape[4]):
        for tseg in range(multi_echo_shots):
            comp[:,:,:,tseg::multi_echo_shots,meas] = comp0[:,:,:,:,meas*multi_echo_shots+tseg]
    print("reshaped to comp")

    Q_after_freqcorr = np.abs(np.sum(comp, axis=-1)) / rms

    #file = os.path.join(os.getcwd(), 'tmp_7_PhaseCorr.nii.gz')
    #nii = nib.Nifti1Image(np.angle(comp).astype(np.float32),affine=np.eye(4))
    #nii.set_data_dtype(np.float32)
    #nib.save(nii, file)
    #


    if TE is None:
        # The phase offsets per time point (const. with respect to TE) are
        # corrected for in the subsequent (old) phase correction step

        # OLD
        comp_roll = np.rollaxis(comp, axis=comp.ndim-1, start=0)

        # phase difference
        hip_roll = hip(np.mean(comp, axis=-1), comp_roll)
        hip_ = np.rollaxis(hip_roll, axis=0, start=comp.ndim)


        gauss_sigma = np.array([2, 2, 2, 0, 0])
        # We should ignore the magnitude of the HiP at this stage, shouldn't we?
        #hip_ = np.angle(hip_)
        #hip_real = gaussian_filter(np.cos(hip_), gauss_sigma)
        #hip_imag = gaussian_filter(np.sin(hip_), gauss_sigma)

        hip_real = gaussian_filter(np.real(hip_), gauss_sigma)
        hip_imag = gaussian_filter(np.imag(hip_), gauss_sigma)

        # hip_smooth = hip_real + 1.0j * hip_imag
        # we don't need the magnitude
        hip_smooth = np.exp(1.0j * np.arctan2(hip_imag, hip_real))

        comp_corr = comp.copy() * hip_smooth
    else:
        comp_corr = comp

    # NEW?

    #hip_ = np.zeros(comp_roll.shape, dtype=float)
    #hip_[1:,...] = hip_phase(np.angle(comp_roll[1:,...]), np.angle(comp_roll[:-1,...]))
    #nib.save(nib.Nifti1Image(hip_, None), os.path.join(os.getcwd(),'pc_phasediff_0.nii.gz'))
    #del comp_roll

    #hip_real = gaussian_filter(np.cos(hip_), [1,2,2,2,0])
    #hip_imag = gaussian_filter(np.sin(hip_), [1,2,2,2,0])
    #hip_ = np.arctan2(hip_imag, hip_real)
    #nib.save(nib.Nifti1Image(hip_, None), os.path.join(os.getcwd(),'pc_phasediff_1.nii.gz'))

    #hip_ = np.unwrap(hip_, axis=0)
    #hip_ = np.cumsum(hip_, axis=0)
    #nib.save(nib.Nifti1Image(hip_, None), os.path.join(os.getcwd(),'pc_phasediff_2.nii.gz'))
    #hip_ -= np.mean(hip_, axis=0)
    #nib.save(nib.Nifti1Image(hip_, None), os.path.join(os.getcwd(),'pc_phasediff_3.nii.gz'))
    #hip_ = np.rollaxis(hip_, axis=0, start=comp.ndim)
    #hip_ = np.exp(-1.0j * hip_)

    #comp_corr = comp.copy() * hip_

    Q_after = np.abs(np.sum(comp_corr, axis=-1)) / rms

    return comp_corr, Q_before, Q_after_freqcorr, Q_after, rms, freq_times_DeltaTE, freq_times_DeltaTE_corr, freq_change_times_DeltaTE

def get_comp_plots_phase_correction(comp,comp_corr):

    import matplotlib.pyplot as plt
    sl = comp.shape[2]/2
    #print('Generating plots...')
    f,a = plt.subplots(comp.shape[-1]+1, comp.shape[-2], figsize=(10,8))
    for te in range(comp.shape[-2]):
        for meas in range(comp.shape[-1]):
            a[meas][te].imshow(np.angle(comp[...,sl,te,meas]))
        a[-1][te].imshow(np.abs(np.mean(comp[...,sl,te,:], axis=-1)), clim=[0,3000], cmap='gray')

    for ax in a.flatten():
        ax.axis('off')
        comp1_png = os.path.join(os.getcwd(), 'comp1.png')
        plt.savefig(comp1_png, dpi=300)

    #print('Generating plots...')
    f,a = plt.subplots(comp.shape[-1]+1, comp.shape[-2], figsize=(10,8))
    for te in range(comp.shape[-2]):
        for meas in range(comp.shape[-1]):
            a[meas][te].imshow(np.angle(comp_corr[...,sl,te,meas]))
        a[-1][te].imshow(np.abs(np.mean(comp_corr[...,sl,te,:], axis=-1)), clim=[0,3000], cmap='gray')
    for ax in a.flatten():
        ax.axis('off')
        comp2_png = os.path.join(os.getcwd(), 'comp2.png')
        plt.savefig(comp2_png,dpi=300)

    #print('Generating plots...')
    f,a = plt.subplots(comp.shape[-1], comp.shape[-2], figsize=(10,8))
    for meas in range(comp.shape[-1]):
        for te in range(comp.shape[-2]):
            a[meas][te].imshow(np.angle(comp_corr[...,sl,te,meas]*np.conj(comp[...,sl,te,meas])), clim=[-np.pi, np.pi])
        a[meas][te].axis('off')
        comp3_png = os.path.join(os.getcwd(), 'comp3.png')
        plt.savefig(comp3_png,dpi=300)

    return comp1_png,comp2_png, comp3_png

def phase_correction(applytopup_corrected, multi_echoes, header, multi_echo_shots=1, blip_up_down=0, echo_times=None):

    import nibabel as nib
    import numpy as np
    import os
    from qsm_pipeline.CustomFunctions import (get_comp_phase_correction,
                                              get_comp_corr_phase_correction)

    affine, comp = get_comp_phase_correction(applytopup_corrected, multi_echoes)
    print("got comp_phase_correction")

    if blip_up_down==0:
        #fix for inaccuracy in phasecorr step: see Ruediger 21.1.2019 email
        #####
        N = comp.shape[-1]
        # CAUTION: Assumes that in the phase difference between AP and PA measurements,
        # there exists a linear gradient along the readout direction, which may be explained
        # by the fact that for AP the whole k-space trajectory is rotated by 180 deg
        # compared to PA (not only PE inversion).
        readout_pc = comp[...,N//2:]*np.conj(comp[...,:N//2])
        print(comp.shape, readout_pc.shape)
        if N>2:
            # average along multiple measurements per PE direction (if available)
            readout_pc = np.mean(readout_pc, axis=-1)
        print(comp.shape, readout_pc.shape)

        # Ahn and Cho approach to estimating linear phase slopw without unwrapping
        readout_pc = readout_pc[1:,:,:,:] * np.conj(readout_pc[:-1,:,:,:])
        print(comp.shape, readout_pc.shape)
        readout_pc = np.mean(readout_pc)
        print(comp.shape, readout_pc.shape)
        X = np.arange(comp.shape[0])-comp.shape[0]*0.5
        readout_pc = np.exp(0.5j*X*np.angle(readout_pc))
        print(comp.shape, readout_pc.shape)

        # Ahn and Cho phase correction approach (not to remove N/2 ghost,
        # but to remove linear phase gradient between AP and PA measurement first
        # to avoid phase wraps in subsequent estimation of phase drifts caused by
        # subject motion or scanner heating, etc.
        comp = np.rollaxis(comp, axis=0, start=comp.ndim)
        comp[...,:N//2,:] *= readout_pc
        comp[...,N//2:,:] *= np.conj(readout_pc)
        comp = np.rollaxis(comp, axis=comp.ndim-1, start=0)
        #####

    #mean_tu = np.mean(comp, axis=-1)

    #nn# mean_magn_tu_file = os.path.join(os.getcwd(), 'mean_magn_tu.nii.gz')
    #nn# mean_phas_tu_file = os.path.join(os.getcwd(), 'mean_phas_tu.nii.gz')

    #nn# nib.save(nib.Nifti1Image(np.abs(np.mean(comp, axis=-1)), affine), mean_magn_tu_file)
    #nn# nib.save(nib.Nifti1Image(np.angle(np.mean(comp, axis=-1)), affine), mean_phas_tu_file)

    comp_corr, Q_before, Q_after_freqcorr, Q_after, rms, PhasDte1, PhasDte2, PhasDteDt = get_comp_corr_phase_correction(comp, multi_echo_shots, echo_times)
    print("got get_comp_corr_phase_correction")


    #mean_pc = np.mean(comp_corr, axis=-1)

    #nn# mean_magn_pc_file = os.path.join(os.getcwd(), 'mean_magn_pc.nii.gz')
    #nn# mean_phas_pc_file = os.path.join(os.getcwd(), 'mean_phas_pc.nii.gz')

    #as mean_pc = np.mean(comp_corr, axis=-1)
    #nib.save(nib.Nifti1Image(np.abs(mean_pc), affine), mean_magn_pc_file)
    #nib.save(nib.Nifti1Image(np.angle(mean_pc), affine), mean_phas_pc_file)
    #nn# nib.save(nib.Nifti1Image(np.abs(np.mean(comp_corr, axis=-1)), affine), mean_magn_pc_file)
    #nn# nib.save(nib.Nifti1Image(np.angle(np.mean(comp_corr, axis=-1)), affine), mean_phas_pc_file)

    """
    get_comp_plots_phase_correction(comp,comp_corr)
    """
    del comp


    Nmeas = comp_corr.shape[-1]


    pc_AP_magn_file = os.path.join(os.getcwd(), 'pc_AP_magn.nii.gz')
    pc_PA_magn_file = os.path.join(os.getcwd(), 'pc_PA_magn.nii.gz')
    pc_AP_phas_file = os.path.join(os.getcwd(), 'pc_AP_phas.nii.gz')
    pc_PA_phas_file = os.path.join(os.getcwd(), 'pc_PA_phas.nii.gz')
    pc_Q_before_file = os.path.join(os.getcwd(), 'pc_Q_before.nii.gz')
    pc_Q_after_freqcorr_file = os.path.join(os.getcwd(), 'pc_Q_after_freqcorr.nii.gz')
    pc_Q_after_file = os.path.join(os.getcwd(), 'pc_Q_after.nii.gz')
    pc_RMS_file = os.path.join(os.getcwd(), 'pc_RMS.nii.gz')
    pc_PhasDte_orig_file = os.path.join(os.getcwd(), 'pc_PhasDte_orig.nii.gz')
    pc_PhasDte_corr_file = os.path.join(os.getcwd(), 'pc_PhasDte_corr.nii.gz')
    pc_PhasDteDt_file = os.path.join(os.getcwd(), 'pc_PhasDteDt.nii.gz')

    if blip_up_down==1:
        nib.save(nib.Nifti1Image(np.abs(comp_corr), affine ,header=header), pc_AP_magn_file)
        nib.save(nib.Nifti1Image(np.angle(comp_corr), affine ,header=header), pc_AP_phas_file)
    elif blip_up_down==2:
        nib.save(nib.Nifti1Image(np.abs(comp_corr), affine ,header=header), pc_PA_magn_file)
        nib.save(nib.Nifti1Image(np.angle(comp_corr), affine ,header=header), pc_PA_phas_file)
    else:
        nib.save(nib.Nifti1Image(np.abs(comp_corr[...,:Nmeas//2]), affine ,header=header), pc_AP_magn_file)
        nib.save(nib.Nifti1Image(np.angle(comp_corr[...,:Nmeas//2]), affine ,header=header), pc_AP_phas_file)
        nib.save(nib.Nifti1Image(np.abs(comp_corr[...,Nmeas//2:]), affine ,header=header), pc_PA_magn_file)
        nib.save(nib.Nifti1Image(np.angle(comp_corr[...,Nmeas//2:]), affine ,header=header), pc_PA_phas_file)

    nib.save(nib.Nifti1Image(Q_before, affine ,header=header), pc_Q_before_file)
    nib.save(nib.Nifti1Image(Q_after_freqcorr, affine ,header=header), pc_Q_after_freqcorr_file)
    nib.save(nib.Nifti1Image(Q_after, affine ,header=header), pc_Q_after_file)
    nib.save(nib.Nifti1Image(rms, affine ,header=header), pc_RMS_file)
    nib.save(nib.Nifti1Image(PhasDte1, affine ,header=header), pc_PhasDte_orig_file)
    nib.save(nib.Nifti1Image(PhasDte2, affine ,header=header), pc_PhasDte_corr_file)
    nib.save(nib.Nifti1Image(PhasDteDt, affine ,header=header), pc_PhasDteDt_file)

    return os.path.abspath(pc_AP_magn_file), \
           os.path.abspath(pc_PA_magn_file), \
           os.path.abspath(pc_AP_phas_file), \
           os.path.abspath(pc_PA_phas_file), \
           os.path.abspath(pc_Q_before_file), \
           os.path.abspath(pc_Q_after_freqcorr_file), \
           os.path.abspath(pc_Q_after_file), \
           os.path.abspath(pc_RMS_file), \
           os.path.abspath(pc_PhasDte_orig_file), \
           os.path.abspath(pc_PhasDte_corr_file), \
           os.path.abspath(pc_PhasDteDt_file)




def create_topup_datain_file(in_filename):
    """
    create acqparams file for topup
    """

    import os

    #in_filename is just for the workflow connection, not used here

    AP_string = "0 1 0 1\n"
    PA_string = "0 -1 0 1\n"
    out_filename = 'datain'

    with open(out_filename,'w') as ofile:
        ofile.write(AP_string)
        ofile.write(PA_string)

    return os.path.abspath(out_filename)


def comp_mag_phase(mag_ap, phase_ap, mag_pa, phase_pa, multi_echoes, multi_echo_shots, phase_factor, blip_up_down=0):

    import nibabel as nib
    import numpy as np
    import os

    print("%s: blip_up_down=%d" % (__name__, blip_up_down))
    print("%s: mag_ap=%s" % (__name__, mag_ap))
    print("%s: phase_ap=%s" % (__name__, phase_ap))
    print("%s: mag_pa=%s" % (__name__, mag_pa))
    print("%s: phase_pa=%s" % (__name__, phase_pa))
    if blip_up_down==0 or blip_up_down==1:
        magn = nib.load(mag_ap)
        phas = nib.load(phase_ap)
    elif blip_up_down==2:
        magn = nib.load(mag_pa)
        phas = nib.load(phase_pa)
    else:
        raise AssertionError('Neither AP nor PA file names have been defined.')

    affine = magn.affine
    header = magn.header

    magn = magn.get_fdata(caching='unchanged', dtype=np.float32)
    if blip_up_down==0:
        magn = np.append(magn, nib.load(mag_pa).get_fdata(caching='unchanged', dtype=np.float32), axis=-1)

    shape = magn.shape
    print(shape)

    if not np.mod(shape[-1], multi_echoes) == 0:
        raise AssertionError('The entered total number of multi-echoes %d must be a factor of the total number of separate volumes %d' % (multi_echoes, shape[-1]))

    tseg = multi_echo_shots # RS: changed from 2 to 1 for single-TE
    Nmeas = shape[-1] / multi_echoes

    print(shape)
    print('Assuming: %d TEs, a factor %d of which are by segmentation, so there are %d measurements och each TE.' % (multi_echoes, tseg, Nmeas))

    phas = phas.get_fdata(caching='unchanged', dtype=np.float32)*phase_factor
    if blip_up_down==0:
        phas = np.append(phas, nib.load(phase_pa).get_fdata(caching='unchanged', dtype=np.float32)*phase_factor, axis=-1)

    comp = magn * np.exp(1.0j * phas)

    del magn, phas

    newshape = np.array(shape)
    newshape[-1] /= tseg*Nmeas
    newshape=np.append(newshape,-1)
    comp = np.reshape(comp, newshape,order='F')
    newshape = comp.shape
    print(newshape)

    Nvols = newshape[-1]
    Ntes  = newshape[-2]

    rmste = np.sqrt(np.mean(np.abs(comp)**2, axis=-2))

    rmste_file = os.path.join(os.getcwd(), 'rmste_magn.nii.gz')
    rms_file   = os.path.join(os.getcwd(), 'rms_magn.nii.gz')

    print("Save RMSTE", rmste.shape)
    nib.save(nib.Nifti1Image(rmste, affine=affine, header=header), rmste_file)

    rms = np.sqrt(np.mean(rmste**2, axis=-1))
    print("Save RMS", rms.shape)
    nib.save(nib.Nifti1Image(rms, affine=affine, header=header), rms_file)

    real_te_files=[]
    imag_te_files=[]

    for te in range(Ntes):
        real_te_file = os.path.join(os.getcwd(), 'real_te%d.nii.gz'%(te))
        imag_te_file = os.path.join(os.getcwd(), 'imag_te%d.nii.gz'%(te))
        nib.save(nib.Nifti1Image(np.real(comp[...,te,:]), affine=affine, header=header), real_te_file)
        nib.save(nib.Nifti1Image(np.imag(comp[...,te,:]), affine=affine, header=header), imag_te_file)
        real_te_files.append(real_te_file)
        imag_te_files.append(imag_te_file)

    #mean raw
    #nn# mean_raw = np.mean(comp, axis=-1)
    #nn# mean_magn_raw_file = os.path.join(os.getcwd(), 'mean_magn_raw.nii.gz')
    #nn# mean_phas_raw_file = os.path.join(os.getcwd(), 'mean_phas_raw.nii.gz')
    #nn# nib.save(nib.Nifti1Image(np.abs(mean_raw),   affine), mean_magn_raw_file)
    #nn# nib.save(nib.Nifti1Image(np.angle(mean_raw), affine), mean_phas_raw_file)

    #rms raw
    #nn# rms_raw = np.sqrt(np.mean(np.abs(mean_raw)**2, axis=-1))
    #nn# rms_raw_file = os.path.join(os.getcwd(), 'rms_magn_raw.nii.gz')
    #nn# nib.save(nib.Nifti1Image(rms_raw, affine = affine), rms_raw_file)

    #pdiff raw
    #nn# pdiff_raw = np.angle(np.mean(hip(comp[...,1:,:], comp[...,:-1,:]), axis=(-2,-1)))
    #nn# pdiff_raw_file = os.path.join(os.getcwd(), 'pdiff_phas_raw.nii.gz')
    #nn# nib.save(nib.Nifti1Image(pdiff_raw, affine = affine), pdiff_raw_file)

    #save comp and comp_reshaped to pass them to reshape
    #nn# comp_file=os.path.join(os.getcwd(), 'comp.nii.gz')
    #nn# comp_reshaped_file = os.path.join(os.getcwd(), 'comp_reshaped.nii.gz')
    #nn# nib.save(nib.Nifti1Image(comp, affine=affine), comp_file)
    #nn# nib.save(nib.Nifti1Image(comp_reshaped, affine=affine), comp_reshaped_file)

    return Ntes, Nvols, newshape, rmste_file, rms_file,real_te_files, imag_te_files, header

def get_plots_reshape_mc(Nvols,Ntes,reshape,tseg,magn_reshaped_mc,phas_reshaped_mc,rmste_mc,pdiffte_mc,blip_up_down=0):

    newshape = reshape
    Nslc  = newshape[2]
    Nro = int(newshape[1])
    Npe = int(newshape[0])
    img_m = np.zeros([int(Nro*(Ntes+2+float(tseg-1)/tseg)), Npe*Nvols])
    img_p = np.zeros([int(Nro*(Ntes+2+float(tseg-1)/tseg)), Npe*Nvols])

    for vol in range(Nvols):
        for te in range(Ntes+2):
            if te<Ntes:
                off = Nro*np.remainder(vol,tseg)//tseg
                arr_m = magn_reshaped_mc[:,:,Nslc//2,te,vol].copy()
                arr_p = phas_reshaped_mc[:,:,Nslc//2,te,vol].copy()
            else:
                off = Nro*np.remainder(Nvols-1,tseg)//tseg
                if te==Ntes:
                    arr_m = rmste_mc[:,:,Nslc//2,vol].copy()
                    arr_p = pdiffte_mc[:,:,Nslc//2,vol].copy()
                else:
                    arr_m = np.abs(rmste_mc[:,:,Nslc//2,vol] - rmste_mc[:,:,Nslc//2,0])
                    arr_p = hip_phase(pdiffte_mc[:,:,Nslc//2,vol], pdiffte_mc[:,:,Nslc//2, 0])
                arr_m[:,-4:] = np.nan
                arr_p[:,-4:] = np.nan
            img_m[te*Nro+off:(te+1)*Nro+off,vol*Npe:(vol+1)*Npe] = np.rot90(arr_m)
            img_p[te*Nro+off:(te+1)*Nro+off,vol*Npe:(vol+1)*Npe] = np.rot90(arr_p)


    plt.figure(figsize=(Nvols*1.5, (Ntes+2+float(tseg-1)/tseg)*1.5))
    #plt.imshow(img_m, cmap='gray', clim=[0,3000])
    plt.axis('off')
    rmste_magn_mc_png = os.path.join(os.getcwd(), 'rmste_magn_mc.png')
    plt.savefig(rmste_magn_mc_png, dpi=300)

    plt.figure(figsize=(Nvols*1.5, (Ntes+2+float(tseg-1)/tseg)*1.5))
    #plt.imshow(img_p, cmap='gray', clim=[-np.pi, np.pi])
    plt.axis('off')
    pdiff_phas_mc_png = os.path.join(os.getcwd(), 'pdiff_phas_mc.png')
    plt.savefig(pdiff_phas_mc_png, dpi=300)



def do_reshape_mc(Ntes, Nvols, shape, mc_real_te, mc_imag_te, multi_echo_shots,header, blip_up_down=0, av_order=0):
    import os
    import nibabel as nib
    import numpy as np

    if not np.mod(shape[-1], multi_echo_shots) == 0:
        raise AssertionError('The entered total number of multi-echo shots %d must be a factor of the total number of separate volumes %d' % (multi_echo_shots, shape[-1]))

    tseg=multi_echo_shots # RS: changed from 2 to 1

    #nn# comp=nib.load(comp_file).get_fdata()
    #nn# comp_reshaped=nib.load(comp_reshaped_file).get_fdata()



    reshape = np.array(shape)
    #reshape[-1] *= tseg
    #reshape[-2] /= tseg

    shape = np.array(shape)
    shape[-1]/=tseg #from shape
    shape[-2]*=tseg #from shape

    real_mc = np.zeros(shape) #from shape
    imag_mc = np.zeros(shape) #from shape

    magn_reshaped_mc = np.zeros(reshape)
    phas_reshaped_mc = np.zeros(reshape)

    newshape = shape.copy() #from shape
    newshape[-1] = -1
    newshape[-2] = tseg

    affine = nib.load(mc_real_te[0]).affine

    for te in range(Ntes):
        real = nib.load(mc_real_te[te]).get_fdata(caching='unchanged', dtype=np.float32)
        imag = nib.load(mc_imag_te[te]).get_fdata(caching='unchanged', dtype=np.float32)
        if av_order==0:
            real_mc[...,te*tseg:(te+1)*tseg,:] = np.reshape(real, newshape, order='F')
            imag_mc[...,te*tseg:(te+1)*tseg,:] = np.reshape(imag, newshape, order='F')
        else:
            Nmeas = real_mc.shape[-1] # repeated measurements with the same contrast (with or without RO or PE polarity change)
            Npe = 1
            if blip_up_down==0:
                Npe = 2
            N = Nmeas//Npe
            print(real.shape, Nmeas, Npe)
            for n in range(Npe): # AP and PA
                for m in range(N): # short-averaging repeated measurements
                    real_mc[...,te*tseg:(te+1)*tseg,n*N+m] = real[...,n*N*tseg+m:(n+1)*N*tseg:N]
                    imag_mc[...,te*tseg:(te+1)*tseg,n*N+m] = imag[...,n*N*tseg+m:(n+1)*N*tseg:N]

        magn_reshaped_mc[...,te,:] = np.sqrt(real**2 + imag**2)
        phas_reshaped_mc[...,te,:] = np.arctan2(imag, real)

    Nmeas = shape[-1] #from shape`
    real_mc_AP = os.path.join(os.getcwd(), 'real_mc_AP.nii.gz')
    real_mc_PA = os.path.join(os.getcwd(), 'real_mc_PA.nii.gz')
    imag_mc_AP = os.path.join(os.getcwd(), 'imag_mc_AP.nii.gz')
    imag_mc_PA = os.path.join(os.getcwd(), 'imag_mc_PA.nii.gz')

    newshape = real_mc.shape[:3] + tuple([-1])
    print("av_order: %i"  %av_order)

    nib.save(nib.Nifti1Image(np.reshape(real_mc[...,:Nmeas//2], newshape, order='F'), affine, header=header), real_mc_AP)
    nib.save(nib.Nifti1Image(np.reshape(real_mc[...,Nmeas//2:], newshape, order='F'), affine, header=header), real_mc_PA)
    nib.save(nib.Nifti1Image(np.reshape(imag_mc[...,:Nmeas//2], newshape, order='F'), affine, header=header), imag_mc_AP)
    nib.save(nib.Nifti1Image(np.reshape(imag_mc[...,Nmeas//2:], newshape, order='F'), affine, header=header), imag_mc_PA)

    real_mc_AP = os.path.abspath(real_mc_AP)
    real_mc_PA = os.path.abspath(real_mc_PA)
    imag_mc_AP = os.path.abspath(imag_mc_AP)
    imag_mc_PA = os.path.abspath(imag_mc_PA)

    if blip_up_down==0:
        rmste_mc = np.sqrt(np.mean(magn_reshaped_mc**2, axis=-2))
        #AP = np.sqrt(np.mean(rmste_mc[...,:Nvols//2]**2, axis=-1))
        #PA = np.sqrt(np.mean(rmste_mc[...,Nvols//2:]**2, axis=-1))
        AP = rmste_mc[...,0]
        PA = rmste_mc[...,Nvols//2]
        APPA_mc = np.append(AP[...,np.newaxis], PA[...,np.newaxis], axis=-1)
        appa_mc_file = os.path.join(os.getcwd(), 'APPA_mc.nii.gz')
        nib.save(nib.Nifti1Image(APPA_mc, affine, header=header), appa_mc_file)

        appa_mc_file = os.path.abspath(appa_mc_file)
    else:
        appa_mc_file = None

    return appa_mc_file, real_mc_AP,real_mc_PA,imag_mc_AP,imag_mc_PA



def moco_iterator(rms_magn, rmste_magn, it):
    import os
    import nibabel as nib
    import numpy as np
    from nipype.interfaces import fsl
    from nipype.utils.filemanip import copyfile

    fn, ext = os.path.splitext(rms_magn)
    ref_file=fn.replace('.nii','')+'_mcflirtref0.nii'+ext
    #ref_file = rms_magn

    copyfile(rms_magn,ref_file,False,use_hardlink=True)

    #out_files=[]
    ref_files=[]
    out_par_file="" #we only want to keep the last iteration
    out_rms_files=[] #we only want to keep the last iteration rms


    out_mat_dirs=[]
    #last out_mat_dir in (0,1,2) is 2 as in applyxfm4D code mcflirt_ref%d.nii.gz (%mcf_iterations-1)
    #is used,and that means the last transform .mat dir is having no.2 in its name
    last_mat_dir=''

    out_file=''

    for i in range(it):

        out_file = 'rmste_magn_mcflirt%d.nii.gz'%(i)

        mcflirt=fsl.MCFLIRT()
        mcflirt.inputs.in_file = rmste_magn
        mcflirt.inputs.ref_file = ref_file #actually the input rms_magn the first time
        mcflirt.inputs.out_file = out_file
        mcflirt.inputs.save_rms = True
        mcflirt.inputs.save_plots = True
        mcflirt.inputs.save_mats=True
        mcflirt.inputs.interpolation='spline' #-spline_final
        # mcflirt.inputs.terminal_output='file' # deprecated in later nipype version

        print(mcflirt.cmdline)

        runtime=mcflirt.run()
        outputs = runtime.outputs.get()
        out_mcf_file = outputs['out_file']

        out_par_file = outputs['par_file']
        out_rms_file = outputs['rms_files']

        out_mat_dir = out_file+'.mat'
        last_mat_dir=os.path.abspath(out_mat_dir) #overwrite until last (2)
        out_mcf_img = nib.load(out_mcf_file)
        affine=out_mcf_img.affine
        header=out_mcf_img.header
        out_mcf_img = out_mcf_img.get_fdata(caching='unchanged', dtype=np.float32)
        ref_file = 'mcflirt_ref%d.nii.gz'%(i+1)
        nib.save(nib.Nifti1Image(np.sqrt(np.mean(out_mcf_img**2, axis=-1)), affine, header=header), ref_file)

        #out_files.append(os.path.abspath(out_file))

        ref_files.append(os.path.abspath(ref_file))
        out_mat_dirs.append(os.path.abspath(out_mat_dir))

        out_file = out_mcf_file
        out_rms_files=[]
        for rms_file in out_rms_file:
            out_rms_files.append(rms_file)


    return out_file, ref_files, out_mat_dirs, last_mat_dir, out_par_file, out_rms_files
