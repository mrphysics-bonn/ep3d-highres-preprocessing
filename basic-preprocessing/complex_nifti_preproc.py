#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :   complex_nifti_preproc.py
@Time    :   2023/06/22 13:29:01
@Author  :   Rüdiger Stirnberg 
@Contact :   ruediger.stirnberg@dzne.de
'''

import os
import sys
import argparse
import glob
from os.path import join as pjoin
import nipype.interfaces.fsl as fsl
import nipype.interfaces.utility as util     # utility
import nipype.pipeline.engine as pe          # pypeline engine
import nipype.interfaces.io as nio
from nipype import SelectFiles
import nibabel as nib
import numpy as np
import time
from datetime import timedelta
from complex_nifti_tools.nipype_helpers import phasematch_cartesian_function, mean_cartesian_function, denoise_cartesian_function
from complex_nifti_tools.nipype_workflows import moco_complex
from complex_nifti_tools.complex_nifti_convert import complex_nifti_convert
import multiprocessing
num_cores = multiprocessing.cpu_count()

def preproc_complex(base_dir=os.getcwd(), name="preproc_complex"):
    preproc_complex = pe.Workflow(name=name)
    preproc_complex.base_dir = base_dir

    # Set up a node to define all inputs required for this workflow
    inputnode = pe.Node(interface=util.IdentityInterface(fields=['magn_file', 'phas_file']), name='inputnode')

    # Scale phase to radian (if necessary) and convert magnitude/phase to real/imag
    pol2cart = pe.Node(interface=util.Function(input_names = ['magn_file', 'phas_file', 'complex_cartesian'],
                                               output_names = ['out_file1', 'out_file2'],
                                               function = complex_nifti_convert),
                                               name='pol2cart')
    pol2cart.inputs.complex_cartesian = 'out'
    preproc_complex.connect(inputnode       , 'magn_file'           , pol2cart      , 'magn_file'           )
    preproc_complex.connect(inputnode       , 'phas_file'           , pol2cart      , 'phas_file'           )
    

    # Motion correction
    moco_flow = moco_complex(preproc_complex.base_dir)
    preproc_complex.connect(inputnode       , 'magn_file'            ,  moco_flow   , 'inputnode.magn_file' )
    preproc_complex.connect(pol2cart        , 'out_file1'            ,  moco_flow   , 'inputnode.real_file' )
    preproc_complex.connect(pol2cart        , 'out_file2'            ,  moco_flow   , 'inputnode.imag_file' )

    # Phase match
    phasematch = pe.Node(interface=util.Function(input_names = ['real_file', 'imag_file'],
                                                 output_names = ['real_out_file', 'imag_out_file'],
                                                 function = phasematch_cartesian_function),
                                                 name='phasematch')

    preproc_complex.connect(moco_flow       , 'outputnode.real_file' , phasematch   , 'real_file')
    preproc_complex.connect(moco_flow       , 'outputnode.imag_file' , phasematch   , 'imag_file')

    # Mean
    mean = pe.Node(interface=util.Function(input_names = ['real_file', 'imag_file'],
                                           output_names = ['real_out_file', 'imag_out_file'],
                                           function = mean_cartesian_function),
                                           name='mean')
    preproc_complex.connect(phasematch      , 'real_out_file'        , mean      , 'real_file' )
    preproc_complex.connect(phasematch      , 'imag_out_file'        , mean      , 'imag_file' )

    # NLM denoise
    denoise = pe.Node(interface=util.Function(input_names = ['real_file', 'imag_file'],
                                              output_names = ['real_out_file', 'imag_out_file'],
                                              function = denoise_cartesian_function),
                                              name='denoise')
    preproc_complex.connect(mean            , 'real_out_file'        , denoise   , 'real_file' )
    preproc_complex.connect(mean            , 'imag_out_file'        , denoise   , 'imag_file' )

    # Convert back to  magn/phas
    cart2pol = pe.Node(interface=util.Function(input_names = ['magn_file', 'phas_file', 'phase_scale', 'complex_cartesian'],
                                               output_names = ['out_file1', 'out_file2'],
                                               function = complex_nifti_convert),
                                               name='cart2pol')
    cart2pol.inputs.complex_cartesian = 'in'
    preproc_complex.connect(mean            , 'real_out_file'       , cart2pol      , 'magn_file'           )
    preproc_complex.connect(mean            , 'imag_out_file'       , cart2pol      , 'phas_file'           )


    cart2pol_denoise = pe.Node(interface=util.Function(input_names = ['magn_file', 'phas_file', 'phase_scale', 'complex_cartesian'],
                                               output_names = ['out_file1', 'out_file2'],
                                               function = complex_nifti_convert),
                                               name='cart2pol_denoise')
    cart2pol_denoise.inputs.complex_cartesian = 'in'
    preproc_complex.connect(denoise         , 'real_out_file'       , cart2pol_denoise, 'magn_file'           )
    preproc_complex.connect(denoise         , 'imag_out_file'       , cart2pol_denoise, 'phas_file'           )

    # Set up a node to define all outputs of this workflow
    outputnode = pe.Node(interface=util.IdentityInterface(fields=['mat_file', 'mean_magn_file', 'mean_phas_file', 'denoised_magn_file', 'denoised_phas_file']), name="outputnode")
    preproc_complex.connect(moco_flow       , 'outputnode.mat_file'  , outputnode, 'mat_file' )
    preproc_complex.connect(cart2pol        , 'out_file1'            , outputnode, 'mean_magn_file' )
    preproc_complex.connect(cart2pol        , 'out_file2'            , outputnode, 'mean_phas_file' )
    preproc_complex.connect(cart2pol_denoise, 'out_file1'            , outputnode, 'denoised_magn_file' )
    preproc_complex.connect(cart2pol_denoise, 'out_file2'            , outputnode, 'denoised_phas_file' )    

    return preproc_complex

def main():
    parser = argparse.ArgumentParser(description='Perform basic complex-valued nifti preprocessing for multiple averages of rapid,'
                                     'single-TE 3D-EPI based GRE acquisitions (e.g. for QSM).',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--magn_and_phas_file', '-m', help='1 magnitude and 1 phase nifti filename', nargs=2)
    group.add_argument('--base_dir_subjects_template', '-b', help='1 base directory including one or more subject folders to be specified, and finally 1 template string', nargs='+', default=None)
    parser.add_argument('--graph-only', '-g', action='store_true', help='only plot graph of the workflow')
    parser.add_argument('--parallel-procs', '-p', type=int, default=num_cores, help='Number of cores for parallel processing. Default=max. available procesors.')
    args = parser.parse_args()

    main_flow = pe.Workflow(name='complex_nifti_preproc')
    main_flow.config['execution'] = {'crashdump_dir': main_flow.base_dir, 'remove_unnecessary_outputs': True} #, 'remove_node_directories': True}


    # Data sink
    datasink = pe.Node(interface=nio.DataSink(), name='datasink')
    datasink.inputs.parameterization = False

    if args.base_dir_subjects_template==None:
        main_flow.base_dir=os.getcwd()
        templates={"magn": args.magn_and_phas_file[0], "phas": args.magn_and_phas_file[1]}
        dg = pe.Node(SelectFiles(templates), "selectfiles")
        dg.inputs.base_directory = main_flow.base_dir

        # Run preproc workflow
        preproc_flow = preproc_complex(base_dir=main_flow.base_dir)
        main_flow.connect(dg    , 'magn'    , preproc_flow, 'inputnode.magn_file')
        main_flow.connect(dg    , 'phas'    , preproc_flow, 'inputnode.phas_file')
    else:
        assert len(args.base_dir_subjects_template)>2, "base_dir_subjects_template argument must be a list of at least 3 strings [base_dir, subject(s), template]."
        base_dir = os.path.abspath(args.base_dir_subjects_template[0])
        templates={'ep3d': pjoin('{subject_id}', args.base_dir_subjects_template[-1])}
        subject_ids = args.base_dir_subjects_template[1:-1]
        dginputnode = pe.Node(interface=util.IdentityInterface(fields=['subj_ids']), name='dginputnode')
        dginputnode.iterables = [('subj_ids', subject_ids)]
        dg = pe.Node(SelectFiles(templates), "selectfiles")
        dg.inputs.base_directory = base_dir
        select_magn = pe.Node(util.Select(), "select_magn")
        select_magn.inputs.index = 0
        select_phas = pe.Node(util.Select(), "select_phas")
        select_phas.inputs.index = 1
        main_flow.base_dir = base_dir
        main_flow.connect(dginputnode   , 'subj_ids'    , dg, 'subject_id')
        main_flow.connect(dg            , 'ep3d'    , select_magn   , 'inlist')
        main_flow.connect(dg            , 'ep3d'    , select_phas   , 'inlist')

        # Run preproc workflow
        preproc_flow = preproc_complex(base_dir=main_flow.base_dir)
        main_flow.connect(select_magn   , 'out' , preproc_flow, 'inputnode.magn_file')
        main_flow.connect(select_phas   , 'out' , preproc_flow, 'inputnode.phas_file')

        main_flow.connect(dginputnode   , 'subj_ids',     datasink,    'container')

    datasink.inputs.base_directory = pjoin(main_flow.base_dir, main_flow.name+'_Results')
    main_flow.connect(preproc_flow   , 'outputnode.mean_magn_file'      , datasink, 'mean.@magn' )
    main_flow.connect(preproc_flow   , 'outputnode.mean_phas_file'      , datasink, 'mean.@phas' )
    main_flow.connect(preproc_flow   , 'outputnode.denoised_magn_file'  , datasink, 'denoised.@magn' )
    main_flow.connect(preproc_flow   , 'outputnode.denoised_phas_file'  , datasink, 'denoised.@phas' )


    # Time everything
    start_time = time.time()


    
    #main_flow.run()
    if args.graph_only:
        main_flow.write_graph()
    else:
        main_flow.run(plugin='MultiProc', plugin_args={'n_procs' : args.parallel_procs})
        #main_flow.run()
    delta = time.time() - start_time
    print("->DONE in %2.f seconds (%s HMS)" % (delta, timedelta(seconds=delta)))

# Main
if (__name__ == "__main__"):
    sys.exit(main())
