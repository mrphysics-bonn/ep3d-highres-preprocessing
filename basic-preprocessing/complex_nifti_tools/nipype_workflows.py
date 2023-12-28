#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :   nipype_workflows.py
@Time    :   2023/06/23 12:52:16
@Author  :   Rüdiger Stirnberg 
@Contact :   ruediger.stirnberg@dzne.de
'''
import os
import nipype.interfaces.fsl as fsl
import nipype.interfaces.utility as util     # utility
import nipype.pipeline.engine as pe          # pypeline engine

def applyxfm4d(base_dir=os.getcwd(), name='applyxfm4d'):
    applyxfm4d = pe.Workflow(name=name)
    applyxfm4d.base_dir = base_dir

    # Set up a node to define all inputs required for this workflow
    inputnode = pe.Node(interface=util.IdentityInterface(fields=['in_file', 'mat_files']), name='inputnode')

    # Split time series into list of 3D images
    split = pe.Node(interface=fsl.utils.Split(), name='split')
    split.inputs.dimension = 't'

    applyxfm4d.connect(inputnode  , 'in_file'         , split         , 'in_file'         )

    # Apply motion correction to real and imag files
    applyxfm = pe.MapNode(interface=fsl.preprocess.ApplyXFM(), name='applyxfm', iterfield=['in_file', 'in_matrix_file'])
    applyxfm.inputs.apply_xfm = True
    applyxfm.inputs.interp = 'sinc'

    applyxfm4d.connect(inputnode  , 'in_file'         , applyxfm      , 'reference'       )
    applyxfm4d.connect(split      , 'out_files'       , applyxfm      , 'in_file'         )
    applyxfm4d.connect(inputnode  , 'mat_files'       , applyxfm      , 'in_matrix_file'  )

    # Merge back to 4D series
    merge = pe.Node(interface=fsl.utils.Merge(), name='merge')
    merge.inputs.dimension = 't'
    applyxfm4d.connect(applyxfm   , 'out_file'        , merge         , 'in_files'        )

    # Rename because output of applyxfm does not contain any previous file name information
    rename = pe.Node(interface=util.Rename(), name='rename')
    rename.inputs.format_string = name
    rename.inputs.keep_ext = True
    applyxfm4d.connect(merge      , 'merged_file'     , rename        , 'in_file'         )

    # Set up a node to define all outputs of this workflow
    outputnode = pe.Node(interface=util.IdentityInterface(fields=['out_file']), name="outputnode")
    applyxfm4d.connect(rename     , 'out_file'        , outputnode    , 'out_file'        )

    return applyxfm4d


def moco_complex(base_dir=os.getcwd(), name='moco_complex'):
    moco_complex = pe.Workflow(name=name)
    moco_complex.base_dir = base_dir

    # Set up a node to define all inputs required for this workflow
    inputnode = pe.Node(interface=util.IdentityInterface(fields=['magn_file', 'real_file', 'imag_file']), name='inputnode')

    # Estimate motion matrices on magnitude
    mcflirt = pe.Node(interface=fsl.preprocess.MCFLIRT(), name='mcflirt')
    mcflirt.inputs.ref_vol = 0
    mcflirt.inputs.save_mats = True
    mcflirt.inputs.cost = 'mutualinfo'

    moco_complex.connect(inputnode  , 'magn_file'   , mcflirt   , 'in_file')

    # Apply motion correction on real and imag
    applyxfm4d_flow_real = applyxfm4d(moco_complex.base_dir, name='applyxfm4d_real')
    applyxfm4d_flow_imag = applyxfm4d(moco_complex.base_dir, name='applyxfm4d_imag')
    moco_complex.connect(inputnode          , 'real_file'   , applyxfm4d_flow_real  , 'inputnode.in_file'     )
    moco_complex.connect(mcflirt            , 'mat_file'    , applyxfm4d_flow_real  , 'inputnode.mat_files'   )
    moco_complex.connect(inputnode          , 'imag_file'   , applyxfm4d_flow_imag  , 'inputnode.in_file'     )
    moco_complex.connect(mcflirt            , 'mat_file'    , applyxfm4d_flow_imag  , 'inputnode.mat_files'   )


    # Set up a node to define all outputs of this workflow
    outputnode = pe.Node(interface=util.IdentityInterface(fields=['real_file', 'imag_file', 'mat_file']), name="outputnode")
    moco_complex.connect(applyxfm4d_flow_real   , 'outputnode.out_file' , outputnode   , 'real_file' )
    moco_complex.connect(applyxfm4d_flow_imag   , 'outputnode.out_file' , outputnode   , 'imag_file' )
    moco_complex.connect(mcflirt                , 'mat_file'            , outputnode   , 'mat_file' )

    return moco_complex