# -*- coding: utf-8 -*-
"""A nipype pre-processing pipeline for single-TE EPI-based QSM data acquired for AVIADHD.

Created on Fri Jun 7 2019

@author: stirnbergr
"""

from __future__ import division
import nipype.pipeline.engine as pe
from nipype import SelectFiles

import nipype.interfaces.fsl as fsl
from nipype import IdentityInterface, DataSink
from nipype.interfaces.utility import Function, Merge
import numpy as np

from .CustomFunctions import (comp_mag_phase, do_reshape_mc, moco_iterator,
                              phase_correction, kspace_combine, complex_average, create_topup_datain_file, contrast_combine)

from .CustomInterfaces import ApplyXFM4D, MTOPUP


def create_qsmepiwf(data_dir, subj_ids, nthreads, multi_echoes, multi_echo_shots, phase_factor, blip_up_down=0, av_order=0, echo_times=None, name='QSM_WF'):

    QsmWF = None

    if blip_up_down==1:
        templates = {"AP": "{subject_id}/*_AP.nii.gz",
                     "AP_Phase": "{subject_id}/*_AP_Phase.nii.gz"}
    elif blip_up_down==2:
        templates = {"PA": "{subject_id}/*_PA.nii.gz",
                     "PA_Phase": "{subject_id}/*_PA_Phase.nii.gz"}
    else:
        templates = {"AP": "{subject_id}/*_AP.nii.gz",
                     "AP_Phase": "{subject_id}/*_AP_Phase.nii.gz",
                     "PA": "{subject_id}/*_PA.nii.gz",
                     "PA_Phase": "{subject_id}/*_PA_Phase.nii.gz"}

    print("av_order: %i"  %av_order)

    QsmWF = get_epi_workflow(templates, data_dir,subj_ids, nthreads, multi_echoes, multi_echo_shots, phase_factor, blip_up_down, av_order, echo_times, name='QsmEpi')

    return QsmWF





def get_epi_workflow(templates, data_dir, subj_ids, nthreads, multi_echoes, multi_echo_shots, phase_factor, blip_up_down=0, av_order=0, echo_times=None, name='QSM_EPI'):
    """
        N: spearate averages (with or without RO alternation) per PE scan (A/P)
        Nte: number of different TEs (via "true" multi-TE and TE segmentation)
        Nteseg: t-segmentation factor
        Nte/Nteseg: number of "true" multi-echoes following same excitation
        Loop order: A/P(2) {outside} averages(N) {outside} tseg(Nteseg)

        Legend:
        !!!: Underlying idea
        []: array shape
        M/P: magnitude/phase part
        I/R: imaginary/real part
        A/D: ascending/descending phase encoding

        Input data is expected as a 4D Nifti dataset where echoes (and averages)
        come in chronological order. The RLS protocol, for example:
        N=2, Nte=6, Ntseg=2 (3 true multi-TEs) ==>
        Anterior-posterior (A encoding): TE 1, 3, 5, 2, 4, 6, 1, 3, 5, 2, 4, 6
        Posterior-anterior (D encoding): TE 1, 3, 5, 2, 4, 6, 1, 3, 5, 2, 4, 6
        each: magnitude and phase.
        In the RLS, dcmstack is used for dicom2nifti conversion using options
        --extract-private --embed-meta to keep complete header information.

        0_prepare
        0.1 Load A/D magnitude and phase data 2(A/D) * 2(M/P) * [x,y,z,Nte*N]
        0.2 merge and reshape M/P from 0.1 according to chronological order
            2(M/P) * [x,y,z,Nte/Nteseg,Nteseg*N*2], i.e. pretend A/D and
            tsegs are like repeated measurements of the same TE, even if
            they are not (but they do correspond to unique timepoints, just like
            repeated measurements)
        0.3 Save corresponging real and imaginary data split by TE, i.e.
            2(R/I) * Nte/Ntseg * [x,y,z,Nteseg*N*2]
        0.4 Compute and save RMS timeseries along (Nte/Nteseg) dimension of M
            from 0.2 [x,y,z,Nteseg*N*2]

        NOTE: Store Niftii header

        1_moco
        !!! Find motion based on RMS(TE) timeseries and apply to corresponding
        !!! real/imag images per TE!!!
        1.1 Apply MCFLIRT on RMS timeseries from 0.4 and iteratively update
            the reference image:
            - initialize reference with original RMS timeseries center image
            - update reference with corrected RMS timeseries center image
        1.2 Loop over multi-echo TEs (Nte/Nteseg) and apply final MCFLIRT MAT affines
            to corresponding real and imag data from 0.4
        1.3 Split real and imag results from 1.2 into A/D measurements, concatenate
            TEs again and store 2(A/D) * 2(R/I) * [x,y,z,Nte,N]
        1.4 Compute RMS along 1st (A) and 2nd half (D) of last dimension of result from
            final MCFLIRT loop of 1.1 2(A/D) * [x,y,z] and store concatenation
            [x,y,z,2].

        2_topup
        2.1 Run topup on result of 1.4
        3_applytopup
        3.1 Loop over real and imaginary parts and PE directions and applytopup
            according to output of 2.1 to corresponding real and imaginary data
            from 1.3

        4_phasecorr (and average within PE direction)
        !!! Determine, for each TE, across measurements the average complex vector per voxel.
        !!! Use that as a reference, with respect to which all measurements are correctedself.
        !!! Smoothing the phase difference between measurements and reference is essential
        !!! in order to not make all measurements identical to the reference upon correction
        !!! and instead keep the actual noise (and motion artifacts) for later averaging.
        4.1 Concatenate real and imag A/D results of 3.1 and compute corresponding
            complex array [x,y,z,Nte,N*2]
        4.2 Compute and store average complex reference image per TE [x,y,z,Nte]
        4.3 Compute Hermetian inner product (HiP) between measurements and reference
            (reflecting the phase difference to the reference):
            - roll measurement dimension forwards [2*N,x,y,z,Nte]
            - compute HiP [2*N,x,y,z,Nte]
            - roll measurement dimension backwards and store R/I parts 2 * [x,y,z,Nte,2*N]
        4.4 Smooth HiP R/I parts along x,y,z and compute complex HiP with unity magnitude
            as a subsequent correction array [x,y,z,Nte,N*2]. Select a sigma of ~2 voxels.
        4.5 Multiply complex array of 4.1 with correction array of 4.4 [x,y,z,Nte,N*2]
        4.6 Average along 1st (A) and 2nd half (D) of last dimension of 4.5 [x,y,z,Nte,2]
        4.7 Store M/P data of 4.6 2(M/P) * [x,y,z,Nte,2]

        5_kcombine (weighted k-space average across PE directions)
        5.1 Load complex data from 4.7 [x,y,z,Nte,2]
        5.2 Store M/P part of average along last dimension as "non-weighted
            combination" 2(M/P) * [x,y,z,Nte]
        5.3 Transform to k-space for A/D separately (fftshift(fftn())) 2(A/D) * [x,y,z,Nte]
        5.4 Estimate appropriate weighting functions for A/D:
            - Average squared k-space magnitude across all but the PE(y) dimensions
              to obtain a 1-d Aw and Dw weighting function 2(A/D) * [y]
            - Normalize weighting to get a 1-d Af and Df filter function
              Af = Aw/(Aw+Dw), Df = Dw/(Aw+Dw)
        5.5 Sum k-space data from 5.3 using filter functions from 5.4:
            - roll y dimension forwards 2(A/D) * [y,x,z,Nte]
            - compute weighted sum [y,x,z,Nte]
            - roll y dimension backwards [x,y,z,Nte]
        5.6 Transform back to image space (ifftn(ifftshift())) and save
            M/P part as "weighted combination" 2(M/P) * [x,y,z,Nte]

        NOTE: combine data with Nifti Header stored above!
    """

    QsmEpi = pe.Workflow(name=name)


    inputnode = pe.Node(interface=IdentityInterface(fields=['subj_ids', 'outputdir']), name='inputnode')
    inputnode.iterables = [('subj_ids', subj_ids)]



    #%% collect outputs
    datasink = pe.Node(interface=DataSink(), name='datasink')
    datasink.inputs.parameterization=False

    QsmEpi.connect(inputnode       , 'subj_ids',     datasink,    'container')
    QsmEpi.connect(inputnode       , 'outputdir',    datasink,    'base_directory')

    fileselector = pe.Node(SelectFiles(templates, sort_filelist=True), name='fileselect')
    fileselector.inputs.base_directory = data_dir

    QsmEpi.connect(inputnode          , 'subj_ids',  fileselector,'subject_id')







    #%% Step 1. moco
    #1.1 compute mag and phase
    compute_mag_phase = pe.Node(interface=Function(input_names=['mag_pa','phase_pa','mag_ap','phase_ap', 'multi_echoes', 'multi_echo_shots', 'phase_factor', 'blip_up_down'],
                                                   output_names=['Ntes','Nvols','newshape', 'rmste_file','rms_file',
                                                                 'real_te_files', 'imag_te_files', 'header'],
                                                   function=comp_mag_phase), name='comp_mag_phase')
    compute_mag_phase.inputs.phase_factor = phase_factor
    compute_mag_phase.inputs.multi_echo_shots = multi_echo_shots
    compute_mag_phase.inputs.multi_echoes = multi_echoes
    compute_mag_phase.inputs.blip_up_down = blip_up_down

    compute_mag_phase.terminal_output = 'allatonce'

    #1.2 moco using mcflirt
    mocoit = pe.Node(interface=Function(input_names=['rms_magn','rmste_magn','it'],
                                            output_names=['out_file','ref_files','out_mat_dirs',
                                                          'last_mat_dir','out_par_file', 'out_rms_files'],
                                            function=moco_iterator), name='moco_iterator')

    mocoit.inputs.it=3

    #1.3 applyxfm4d to real te images
    applyxfm4d_real = pe.MapNode(interface=ApplyXFM4D(),iterfield=['input_volume'], name='applyxfm4d_real')
    applyxfm4d_real.inputs.userprefix='MAT_'
    applyxfm4d_imag = pe.MapNode(interface=ApplyXFM4D(),iterfield=['input_volume'], name='applyxfm4d_imag')
    applyxfm4d_imag.inputs.userprefix='MAT_'

    #1.4 reshape
    reshape_mc = pe.Node(interface=Function(input_names=['Ntes', 'Nvols', 'shape',
                                                             'mc_real_te', 'mc_imag_te', 'multi_echo_shots', 'header', 'blip_up_down','av_order'],
                                            output_names=['appa_mc_file',
                                                          'real_mc_AP','real_mc_PA','imag_mc_AP','imag_mc_PA'],
                                            function=do_reshape_mc), name='reshape_mc')
    reshape_mc.inputs.multi_echo_shots = multi_echo_shots
    reshape_mc.inputs.blip_up_down     = blip_up_down
    reshape_mc.inputs.av_order         = av_order
    reshape_mc.terminal_output = 'allatonce'

    #%% Step 2. topup

    #2.1 create topup acq param file
    create_topup_param_file = pe.Node(interface=Function(input_names=['in_filename'],
                                                             output_names=['out_filename'],
                                                         function=create_topup_datain_file), name='create_topup_param_file')

    #2.2 topup
    topup = pe.Node(interface=MTOPUP(), name = 'topup')
    topup.inputs.output_type= 'NIFTI_GZ'
    topup.inputs.warp_res='20,10,4'
    topup.inputs.subsamp='2,1,1'
    topup.inputs.fwhm='8,3,0'
    topup.inputs.max_iter='3,3,12'
    topup.inputs.reg_lambda='0.005,0.000005,0.00000000001'
    topup.inputs.ssqlambda=1
    topup.inputs.regmod='bending_energy'
    topup.inputs.estmov='0,0,0'
    topup.inputs.minmet='1,1,1'
    topup.inputs.splineorder=3
    topup.inputs.interp='spline'
    topup.inputs.scale=1
    topup.inputs.numprec='float'
    topup.inputs.out_base='topup'
    topup.inputs.out_corrected='topup_check'


    #%% Step 3 apply topup
    #3.0 first merge the 4 outputs in a list

    merge_ri_list = pe.Node(interface=Merge(4),name='ri2list')
    merge_indices_list = pe.Node(interface=Merge(1),name='indices2list')
    merge_indices_list.inputs.in1=[[1],[2],[1],[2]]

    #3.1 apply topup
    applytopup = pe.MapNode(interface=fsl.ApplyTOPUP(), iterfield=['in_files','in_index'], name='applytopup')
    applytopup.inputs.method='jac'

    #%% Step 4 phase corr

    phase_correct = pe.Node(interface=Function(input_names=['applytopup_corrected', 'multi_echoes', 'header', 'multi_echo_shots', 'blip_up_down', 'echo_times'],
                                                  output_names=['pc_AP_magn_file','pc_PA_magn_file',
                                                                'pc_AP_phas_file','pc_PA_phas_file',
                                                                'pc_Q_before_file', 'pc_Q_after_freq_corr_file',
                                                                'pc_Q_after_file', 'pc_RMS_file',
                                                                'pc_FreqDte_orig_file', 'pc_FreqDte_corr_file', 'pc_FreqDteChange_file' ],
                                                  function=phase_correction), name='phase_correction')
    phase_correct.inputs.multi_echoes = multi_echoes
    phase_correct.inputs.multi_echo_shots = multi_echo_shots
    phase_correct.inputs.blip_up_down = blip_up_down
    phase_correct.inputs.echo_times = echo_times

    #%% Step 5 kcombine
    if blip_up_down==0:
        kcombine = pe.Node(interface=Function(input_names=['pc_AP_magn_file','pc_PA_magn_file','pc_AP_phas_file','pc_PA_phas_file','header'],
                                              output_names=['mean_magn_file', 'mean_phas_file','mean_unfilt_file'],
                                              function=kspace_combine),name='kcombine')
    else:
        kcombine = pe.Node(interface=Function(input_names=['magn_file','phas_file','header'],
                                              output_names=['mean_magn_file', 'mean_phas_file','mean_unfilt_file'],
                                              function=complex_average),name='kcombine')

    # %% workflow connections

    ccombine = pe.Node(interface=Function(input_names=['magn_file', 'phas_file', 'multi_echo_shots','header'],
                                                  output_names=['comb_magn_file', 'comb_phas_file', 'rmste_file', 'rms_file'],
                                                  function=contrast_combine), name='contrast_combine')
    ccombine.inputs.multi_echo_shots = multi_echo_shots

    #step 1.1
    if blip_up_down==1:
        QsmEpi.connect(fileselector       , 'AP',               compute_mag_phase, 'mag_ap')
        QsmEpi.connect(fileselector       , 'AP_Phase',         compute_mag_phase, 'phase_ap')
        compute_mag_phase.inputs.mag_pa = ''
        compute_mag_phase.inputs.phase_pa = ''
    elif blip_up_down==2:
        compute_mag_phase.inputs.mag_ap = ''
        compute_mag_phase.inputs.phase_ap = ''
        QsmEpi.connect(fileselector       , 'PA',               compute_mag_phase, 'mag_pa')
        QsmEpi.connect(fileselector       , 'PA_Phase',         compute_mag_phase, 'phase_pa')
    else:
        QsmEpi.connect(fileselector       , 'AP',               compute_mag_phase, 'mag_ap')
        QsmEpi.connect(fileselector       , 'AP_Phase',         compute_mag_phase, 'phase_ap')
        QsmEpi.connect(fileselector       , 'PA',               compute_mag_phase, 'mag_pa')
        QsmEpi.connect(fileselector       , 'PA_Phase',         compute_mag_phase, 'phase_pa')


    #step 1.2
    QsmEpi.connect(compute_mag_phase  ,'rms_file',          mocoit,            'rms_magn')
    QsmEpi.connect(compute_mag_phase  ,'rmste_file',        mocoit,            'rmste_magn')

    #step 1.3
    QsmEpi.connect(compute_mag_phase  ,'rms_file',          applyxfm4d_real,   'ref_volume')
    QsmEpi.connect(compute_mag_phase  ,'real_te_files',     applyxfm4d_real,   'input_volume')
    #QsmEpi.connect(mocoit             ,'out_mat_dirs',      applyxfm4d_real,   'transform_dir')
    QsmEpi.connect(mocoit             ,'last_mat_dir',      applyxfm4d_real,   'transform_dir') #the last mat dir with num 2 in its name

    QsmEpi.connect(compute_mag_phase  ,'rms_file',          applyxfm4d_imag,   'ref_volume')
    QsmEpi.connect(compute_mag_phase  ,'imag_te_files',     applyxfm4d_imag,   'input_volume')
    #QsmEpi.connect(mocoit             ,'out_mat_dirs',      applyxfm4d_imag,   'transform_dir')
    QsmEpi.connect(mocoit             ,'last_mat_dir',      applyxfm4d_imag,   'transform_dir') #the last mat dir with num 2 in its name

    # Send original header to functions
    QsmEpi.connect(compute_mag_phase, 'header', phase_correct, 'header')
    QsmEpi.connect(compute_mag_phase, 'header', reshape_mc,    'header')
    QsmEpi.connect(compute_mag_phase, 'header', kcombine,      'header')
    QsmEpi.connect(compute_mag_phase, 'header', ccombine,      'header')

    #step 1.4
    QsmEpi.connect(compute_mag_phase  ,'Ntes',              reshape_mc,        'Ntes')
    QsmEpi.connect(compute_mag_phase  ,'Nvols',             reshape_mc,        'Nvols')
    QsmEpi.connect(compute_mag_phase  ,'newshape',          reshape_mc,        'shape')
    QsmEpi.connect(applyxfm4d_real    ,'output_volume',     reshape_mc,        'mc_real_te')
    QsmEpi.connect(applyxfm4d_imag    ,'output_volume',     reshape_mc,        'mc_imag_te')

    #step 3.0
    QsmEpi.connect(reshape_mc        ,'real_mc_AP',         merge_ri_list,     'in1')
    QsmEpi.connect(reshape_mc        ,'real_mc_PA',         merge_ri_list,     'in2')
    QsmEpi.connect(reshape_mc        ,'imag_mc_AP',         merge_ri_list,     'in3')
    QsmEpi.connect(reshape_mc        ,'imag_mc_PA',         merge_ri_list,     'in4')

    if blip_up_down == 0:
        #step 2.1
        QsmEpi.connect(reshape_mc         ,'appa_mc_file',      create_topup_param_file,'in_filename')
        #step 2.2
        QsmEpi.connect(reshape_mc         ,'appa_mc_file',      topup,             'in_file')
        QsmEpi.connect(create_topup_param_file,'out_filename',  topup,             'encoding_file')



        #step 3.1
        QsmEpi.connect(merge_ri_list     ,'out',                applytopup,        'in_files')
        QsmEpi.connect(create_topup_param_file,'out_filename',  applytopup,        'encoding_file')
        QsmEpi.connect(topup             ,'out_fieldcoef',      applytopup,        'in_topup_fieldcoef')
        QsmEpi.connect(topup             ,'out_movpar',         applytopup,        'in_topup_movpar')
        QsmEpi.connect(merge_indices_list,'out',                applytopup,        'in_index')

        #step4
        QsmEpi.connect(applytopup         ,'out_corrected',     phase_correct,     'applytopup_corrected')

        #step5
        #'pc_AP_magn_file','pc_PA_magn_file','pc_AP_phas_file','pc_PA_phas_file'
        QsmEpi.connect(phase_correct      ,'pc_AP_magn_file',   kcombine,          'pc_AP_magn_file')
        QsmEpi.connect(phase_correct      ,'pc_PA_magn_file',   kcombine,          'pc_PA_magn_file')
        QsmEpi.connect(phase_correct      ,'pc_AP_phas_file',   kcombine,          'pc_AP_phas_file')
        QsmEpi.connect(phase_correct      ,'pc_PA_phas_file',   kcombine,          'pc_PA_phas_file')

    else:
        QsmEpi.connect(merge_ri_list      ,'out'          ,     phase_correct,     'applytopup_corrected')
        if blip_up_down==1:
            QsmEpi.connect(phase_correct  ,'pc_AP_magn_file',   kcombine,          'magn_file')
            QsmEpi.connect(phase_correct  ,'pc_AP_phas_file',   kcombine,          'phas_file')
            QsmEpi.connect(phase_correct  ,'pc_AP_magn_file',   datasink,          'phasecorr.@magn_file')
            QsmEpi.connect(phase_correct  ,'pc_AP_phas_file',   datasink,          'phasecorr.@phas_file')
        else:
            QsmEpi.connect(phase_correct  ,'pc_PA_magn_file',   kcombine,          'magn_file')
            QsmEpi.connect(phase_correct  ,'pc_PA_phas_file',   kcombine,          'phas_file')
            QsmEpi.connect(phase_correct  ,'pc_PA_magn_file',   datasink,          'phasecorr.@magn_file')
            QsmEpi.connect(phase_correct  ,'pc_PA_phas_file',   datasink,          'phasecorr.@phas_file')

    QsmEpi.connect(kcombine           , 'mean_magn_file'         ,   ccombine,   'magn_file')
    QsmEpi.connect(kcombine           , 'mean_phas_file'         ,   ccombine,   'phas_file')

    # outputs
    QsmEpi.connect(phase_correct      , 'pc_Q_before_file',datasink,   'phasecorr.@Q_before')
    QsmEpi.connect(phase_correct      , 'pc_Q_after_freq_corr_file',datasink,'phasecorr.@Q_after_freq_corr')
    QsmEpi.connect(phase_correct      , 'pc_Q_after_file' ,datasink,   'phasecorr.@Q_after')
    QsmEpi.connect(phase_correct      , 'pc_RMS_file'     ,datasink,   'phasecorr.@RMS')
    QsmEpi.connect(phase_correct      , 'pc_FreqDte_orig_file',datasink,   'phasecorr.@Q_FreqDt_orig')
    QsmEpi.connect(phase_correct      , 'pc_FreqDte_corr_file' ,datasink,   'phasecorr.@Q_FreqDt_corr')
    QsmEpi.connect(phase_correct      , 'pc_FreqDteChange_file' ,datasink,   'phasecorr.@Q_FreqDt_change')

    QsmEpi.connect(mocoit             , 'out_file',        datasink,          'moco.@outfile')
    QsmEpi.connect(mocoit             , 'out_par_file',    datasink,          'moco.@outparfile')
    QsmEpi.connect(mocoit             , 'out_rms_files',   datasink,          'moco.@outrmsfiles')

    QsmEpi.connect(kcombine           , 'mean_unfilt_file'  ,datasink,   'kcombine.@unfilt_files')
    QsmEpi.connect(kcombine           , 'mean_magn_file',datasink,   'kcombine.@magnitude_file')
    QsmEpi.connect(kcombine           , 'mean_phas_file',datasink,   'kcombine.@phase_file')

    QsmEpi.connect(ccombine   , 'comb_magn_file', datasink,   'ckombine.@magn_file')
    QsmEpi.connect(ccombine   , 'comb_phas_file', datasink,   'ckombine.@phas_file')
    QsmEpi.connect(ccombine   , 'rmste_file', datasink,   'ckombine.@rmste_file')
    QsmEpi.connect(ccombine   , 'rms_file', datasink,   'ckombine.@rms_file')



    return QsmEpi
