# -*- coding: utf-8 -*-

import nibabel as nib
import numpy as np
import argparse
import sys
from scipy.ndimage import gaussian_filter

TAG = '[CPHS]'

def hip(comp1, comp2):
    hip = comp1 * np.conj(comp2)
    return hip

def complex_nifti_phasematch(magn_file, phas_file, phase_scale, contrasts, averaging='long', echo_times=None, out=None, stage=None):

    print(TAG, 'Load magnitude')
    nii = nib.load(magn_file)
    magn = nii.get_fdata().astype(np.float32)
    print(TAG, 'Load phase')
    phas = nib.load(phas_file).get_fdata().astype(np.float32) * phase_scale

    comp = magn * np.exp(1.0j*phas)
    repeats = comp.shape[-1]
    multi_te = True
    if echo_times is not None:
        if contrasts==0:
            if np.mod(repeats,len(echo_times))==0:
                rephased_echoes = len(echo_times)
            else:
                raise AssertionError('If number of volumes is not divisable by the number of TEs specified (or vice versa), you need to also specify the numbe of multi-contrasts (e.g. the number of averages)')
        else:
            rephased_echoes = len(echo_times)-contrasts+1
        repeats = repeats//rephased_echoes
    else:
        if contrasts==0:
            rephased_echoes = 1
            print(TAG, 'No TEs and no multi-contrasts specified. Assume single-TE acquisition with %d measurements.'%(repeats))
            multi_te = False
        else:
            rephased_echoes = repeats//contrasts
            print(TAG, 'No TEs specified, but multi-contrasts=%d. Assume multi-TE acquisition with %d TEs and no repetition of identical multi-contrasts.'%(contrasts,rephased_echoes))
        repeats = repeats//rephased_echoes

    if contrasts==0:
        contrasts=1
    else:
        repeats = repeats//contrasts
    new_sh = list(magn.shape[:-1]) + [-1] + [repeats*contrasts]
    print(TAG, 'Reshape to new shape:')
    if averaging == 'long':
        order = 'C'
    else:
        order = 'F'

    comp = np.reshape(comp, new_sh, order=order)
    print(TAG, comp.shape)

    new_sh = comp.shape
    comp[~np.isfinite(comp)]=0.0

    TE = np.zeros(new_sh[3:], dtype=float)
    rephased_echoes = new_sh[3]
    #constuct echo time array that corresponds to the comp array (just without spatial dimensions)
    if echo_times is not None:
        echo_times = np.array(echo_times)
        if averaging=='long':
            for meas in range(repeats):
                for con in range(contrasts):
                    TE[:,meas+con::contrasts] = echo_times[:rephased_echoes, np.newaxis]
                    if con>0:
                        TE[:,meas+con::contrasts] += (echo_times[rephased_echoes+con-1] - echo_times[0])
        else:
            for meas in range(repeats):
                for con in range(contrasts):
                    TE[:,con*repeats:(con+1)*repeats] = echo_times[:rephased_echoes, np.newaxis]
                    if con>0:
                        TE[:,con*repeats:(con+1)*repeats] += (echo_times[rephased_echoes+con-1] - echo_times[0])

    if echo_times is not None:
        TE *= 1.0e-3 #ms -> s
        print("constructed actual echo time matrix to be passed to get_comp_corr_frequency_correcion:")
        print(TE)
    else:
        TE = None

    if out is None:
        phas_out_file = phas_file + '.pmatch.nii.gz'
        Qbefore_file = magn_file + '.qbefore.nii.gz'
        Qafter_file = phas_file + '.qafter.nii.gz'
    else:
        phas_out_file = out + '_pmatch.nii.gz'
        Qbefore_file = out + '_qbefore.nii.gz'
        Qafter_file = out + '_qafter.nii.gz'

    new_sh = comp.shape
    comp[~np.isfinite(comp)]=0.0

    rms = np.sqrt(np.mean(np.abs(comp)**2, axis=-1)) * comp.shape[-1]
    nib.save(nib.Nifti1Image(np.abs(rms), nii.affine, nii.header), 'test.nii.gz')

    Q_before = np.abs(np.sum(comp, axis=-1)) / rms
    nib.save(nib.Nifti1Image(Q_before, nii.affine, nii.header), Qbefore_file)

    if not stage or stage==1:
        print(TAG,  'Stage 1: Linear phase match along first dim (RO) between first half and second half of scans '
                    'using Ahn and Cho\'s autocorrelation method (ineffective and harmless, if there was no systematic RO shift).')
        comp = phase_linear_autocorr(comp)

    if not stage or stage==2:
        print(TAG,  'Stage 2: Frequency-change-based phase correction (in parts even useful without TEs specified.')
        if not multi_te:
            print(TAG,  'Skipped because data is presumably single-TE data (otherwise, specify TEs and/or the number of multi-contrasts or repetitions the 4th dimension contains).')
        else:
            comp = get_comp_corr_frequency_correction(comp, TE)  

    if not stage or stage==3:
        print(TAG,  'Stage 3: Final phase matching (if TEs are not specified).')
        comp = get_comp_corr_phase_correction(comp, TE)  

    Q_after = np.abs(np.sum(comp, axis=-1)) / rms
    nib.save(nib.Nifti1Image(Q_after, nii.affine, nii.header), Qafter_file)

    new_sh = list(magn.shape[:-1]) + [-1]
    print(TAG, 'Reshape back:')
    if averaging == 'long':
        order = 'C'
    else:
        order = 'F'
    comp = np.reshape(comp, new_sh, order=order)

    print(TAG, 'Save phase', phas_out_file)
    nib.save(nib.Nifti1Image(np.angle(comp)/phase_scale, nii.affine, nii.header), phas_out_file)

def phase_linear_autocorr(comp):

    import nibabel as nib
    import numpy as np
  

    #fix for inaccuracy in phasecorr step: see Ruediger 21.1.2019 email
    #####
    N = comp.shape[-1]
    # CAUTION: Assumes that in the phase difference between AP and PA measurements,
    # there exists a linear gradient along the readout direction, which may be explained
    # by the fact that for AP the whole k-space trajectory is rotated by 180 deg
    # compared to PA (not only PE inversion).
    readout_pc = comp[...,N//2:]*np.conj(comp[...,:N//2])
    if N>2:
        # average along multiple measurements per PE direction (if available)
        readout_pc = np.mean(readout_pc, axis=-1)

    # Ahn and Cho approach to estimating linear phase slopw without unwrapping
    readout_pc = readout_pc[1:,:,:,:] * np.conj(readout_pc[:-1,:,:,:])
    readout_pc = np.mean(readout_pc)
    X = np.arange(comp.shape[0])-comp.shape[0]*0.5
    readout_pc = np.exp(0.5j*X*np.angle(readout_pc))

    # Ahn and Cho phase correction approach (not to remove N/2 ghost,
    # but to remove linear phase gradient between AP and PA measurement first
    # to avoid phase wraps in subsequent estimation of phase drifts caused by
    # subject motion or scanner heating, etc.
    comp = np.rollaxis(comp, axis=0, start=comp.ndim)
    comp[...,:N//2,:] *= readout_pc
    comp[...,N//2:,:] *= np.conj(readout_pc)
    comp = np.rollaxis(comp, axis=comp.ndim-1, start=0)
    
    return comp


def get_comp_corr_frequency_correction(comp0, TE=None):
    from scipy.ndimage import gaussian_filter
    print("enter get_comp_corr_frequency_correcion")
    # 1. Determine frequency (times Delta TE) per self-contained measurment (t-seg and repetition).
    #    This is a wrapped Delta Phase image (with anatomical structure)
    # 0      | f1 (t2-t1)    | f1 (t3-t2)      | f1 (t4-t3)
    # 0      | f2 (t2-t1)    | f2 (t3-t2)      | f2 (t4-t3)
    # 0      | f3 (t2-t1)    | f3 (t3-t2)      | f3 (t4-t3)
    # ...
    HiP_dPhase_dTE = np.zeros_like(comp0)
    HiP_dPhase_dTE[...,1:,:] = comp0[...,1:,:]*np.conj(comp0[...,:-1,:])

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

        norm = np.sum(weights, axis=-2)

        for m in range(1,dPhase_dTE_dt.shape[-1]): # time points
            for t in range(1,dPhase_dTE_dt.shape[-2]): # echo times
                weights[...,t,m] /= norm[...,m]
                weights[~np.isfinite(weights[...,t,m]),t,m] = 0.0 # avoid nans or infinites in dPhase_dTE_dt below
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

    comp0 *= np.exp(-1.0j * phase)
    print("times phase")

    return comp0

def get_comp_corr_phase_correction(comp, TE):
 
    if TE is None:
        # The phase offsets per time point (const. with respect to TE) are
        # corrected for in the subsequent (old) phase correction step

        # OLD
        comp_roll = np.rollaxis(comp, axis=comp.ndim-1, start=0)

        # phase difference
        hip_roll = hip(np.mean(comp, axis=-1), comp_roll)
        hip_ = np.rollaxis(hip_roll, axis=0, start=comp.ndim)


        gauss_sigma = np.array([2, 2, 2, 0, 0])

        hip_real = gaussian_filter(np.real(hip_), gauss_sigma)
        hip_imag = gaussian_filter(np.imag(hip_), gauss_sigma)

        hip_smooth = np.exp(1.0j * np.arctan2(hip_imag, hip_real))

        comp_corr = comp.copy() * hip_smooth
    else:
        print("Skip this heuristic step, since previous TE-informed correction has already taken care of this.")
        comp_corr = comp

    return comp_corr

def main():
    parser = argparse.ArgumentParser(
    description="Performs phase matching across multi-contrast measurements (repeated or not). "
                "For multi-echo data, this is done by frequency change estimation across "
                "self-contained measurement time points and corresponding phase correction "
                "with respect to the first time point. This wirks best, if TEs are specified. "
                "If TEs are not specified, frequency-based phase matching is still performed, "
                "but it is less accurate, because it involves numerical integration over long-TE "
                "data with ppor SNR. For single-TE data, the specific TE is irrelevant. "
                "In this case, broad phase matching across repeated measurements is done. "
                "The output phase is going to be scaled in accordance with the input phase "
                "scaling (by default -4096...4096, but can be specified). "
                "Supports multiple contrasts and \"short\" and \"short\" averaging "
                "contrast measurements order.")
    parser.add_argument('magn_file', type=str)
    parser.add_argument('phas_file', type=str) 
    parser.add_argument('--out', '-o', type=str, default=None,
    help =      'Specify a common base name for all outputs instead of the input '
                'file names of magnitude and phase as the default basis.')
    parser.add_argument('--scale-phase', '-s', type=float, default=np.pi/4096.0)
    parser.add_argument('--contrasts', '-c', type=int, default=0,
    help =      'If the input contains multiple different contrasts, e.g. multi-TE shots or MPM contrasts')
    parser.add_argument('--averaging', '-a', choices=['short', 'long'],
    help =      'Whether the same multi-contrast repeats before the next contrast (short) '
                'or whether the multi-contrasts permute before being repeated (long).')
    parser.add_argument('-e', '--echo-times', help='Specify actual echo times [ms] in the following order: '
                                                    'All rephased echo times of first multi-contrast '
                                                    'followed by the first echo times of subsequent multi-contrasts. '
                                                    'A complete TE matrix is computed internally based on these values. '
                                                    'A better, informed phase correction can then be performed, in a '
                                                    'single step instead of pseudo freq. correction followed by additional '
                                                    'phase matching.\n'
                                                    'NOTE:\n'
                                                    'This assumes that the rephased echo time '
                                                    'spacings are the same across all multi-contrasts.\n'
                                                    'If not specified, a single-echo acquisition is assumed, '
                                                    'i.e. all repetitions along the 4th dimension are considererd '
                                                    'averages or multi-contrasts with the same TE!'
                                                    '', required=False, default=None, type=float, nargs='+')
    parser.add_argument('--matching-stage', '-m', type=int, choices=[1,2,3], default=None,
                                                help =  "Which stage of the phase matching should be performed "
                                                        "If not specified, all stages are performed, if applicable:\n"
                                                        "Stage 1: Linear phase match along first dim (RO) between first half and second half of scans "
                                                        "using Ahn and Cho\'s autocorrelation method\n"
                                                        "Stage 2: Frequency-change-based phase correction (in parts even useful without TEs specified.)\n"
                                                        "Stage 3: Final phase matching (if TEs are not specified).")

    args = parser.parse_args()


    complex_nifti_phasematch(args.magn_file, args.phas_file, args.scale_phase, args.contrasts, args.averaging, args.echo_times, args.out, args.matching_stage)


if __name__ == '__main__':
    sys.exit(main())
