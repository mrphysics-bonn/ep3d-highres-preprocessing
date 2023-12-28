#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  4 10:41:07 2018

@author: shahidm
"""

from __future__ import (print_function, division)


import os
from nipype.interfaces.base import (traits, TraitedSpec,InputMultiPath,isdefined,File)
from nipype.interfaces.fsl.base import FSLCommand, FSLCommandInputSpec, Info
from nipype.utils.filemanip import split_filename

import numpy as np
import nibabel as nb


class ApplyXFM4DInputSpec(FSLCommandInputSpec):
    """
    usage:
        Usage: applyxfm4D <input volume> <ref volume> <output volume> <transformation matrix file/[dir]> [-singlematrix/-fourdigit/-userprefix <prefix>]]
    """
    input_volume = traits.File(exists=True, desc="input volume",  argstr="%s", position=0)
    ref_volume   = traits.File(exists=True, desc="refnc volume",  argstr="%s", position=1)
    output_volume= traits.File(genfile=True, desc="output volume", argstr="%s", position=2)
    transform_dir= traits.Directory(exists=True, desc="transform files dir", argstr="%s", position=3)
    userprefix   = traits.String(desc="userprefix", argstr="-userprefix %s", position=4)

class ApplyXFM4DOutputSpec(FSLCommandInputSpec):
    output_volume = traits.File(exists=True, desc="output mc volume")

class ApplyXFM4D(FSLCommand):
    """Currently just a light wrapper around applyxfm4D,
    with no modifications
    ApplyXFM4D is used to apply an existing tranform to an image

    """
    _cmd = 'applyxfm4D'

    input_spec =  ApplyXFM4DInputSpec
    output_spec = ApplyXFM4DOutputSpec

    def __init__(self, **inputs):
        return super(ApplyXFM4D, self).__init__(**inputs)


    def _run_interface(self, runtime):

        runtime = super(ApplyXFM4D, self)._run_interface(runtime)
        #self.inputs.out_dir=os.getcwd()
        if runtime.stderr:
           self.raise_exception(runtime)
        return runtime

    def _gen_outfilename(self):
        output_volume = self.inputs.output_volume
        if not isdefined(output_volume) and isdefined(self.inputs.input_volume):
            self.inputs.output_volume = self._gen_fname(self.inputs.input_volume, suffix='_mc')
        return os.path.abspath(self.inputs.output_volume)


    def _gen_filename(self, name):
        if name == 'output_volume':
            return self._gen_outfilename()
        return None

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs['output_volume']         = self.inputs.output_volume

        return outputs


class MTOPUPInputSpec(FSLCommandInputSpec):
    in_file = File(exists=True, mandatory=True,
                   desc='name of 4D file with images', argstr='--imain=%s')
    encoding_file = File(exists=True, mandatory=True,
                         xor=['encoding_direction'],
                         desc='name of text file with PE directions/times',
                         argstr='--datain=%s')
    encoding_direction = traits.List(traits.Enum('y', 'x', 'z', 'x-', 'y-',
                                                 'z-'), mandatory=True,
                                     xor=['encoding_file'],
                                     requires=['readout_times'],
                                     argstr='--datain=%s',
                                     desc=('encoding direction for automatic '
                                           'generation of encoding_file'))
    readout_times = InputMultiPath(traits.Float,
                                   requires=['encoding_direction'],
                                   xor=['encoding_file'], mandatory=True,
                                   desc=('readout times (dwell times by # '
                                         'phase-encode steps minus 1)'))
    out_base = File(desc=('base-name of output files (spline '
                          'coefficients (Hz) and movement parameters)'),
                    name_source=['in_file'], name_template='%s_base',
                    argstr='--out=%s', hash_files=False)
    out_field = File(argstr='--fout=%s', hash_files=False,
                     name_source=['in_file'], name_template='%s_field',
                     desc='name of image file with field (Hz)')
    out_warp_prefix = traits.Str("warpfield", argstr='--dfout=%s', hash_files=False,
                                 desc='prefix for the warpfield images (in mm)',
                                 usedefault=True)
    out_jac_prefix = traits.Str("jac", argstr='--jacout=%s',
                                 hash_files=False,
                                 desc='prefix for the warpfield images',
                                 usedefault=True)
    out_corrected = File(argstr='--iout=%s', hash_files=False,
                         name_source=['in_file'], name_template='%s_corrected',
                         desc='name of 4D image file with unwarped images')
    out_logfile = File(argstr='--logout=%s', desc='name of log-file',
                       name_source=['in_file'], name_template='%s_topup.log',
                       keep_extension=True, hash_files=False)

    # TODO: the following traits admit values separated by commas, one value
    # per registration level inside topup. threfore i changed this float to string
    #to accept comma separated values as one single string
    warp_res =traits.String(argstr='--warpres=%s',
                            desc=('(approximate) resolution (in mm) of warp '
                                  'basis for the different sub-sampling levels'
                                  '.'))
    subsamp = traits.String(argstr='--subsamp=%s',
                         desc='sub-sampling scheme')
    fwhm = traits.String(argstr='--fwhm=%s',
                        desc='FWHM (in mm) of gaussian smoothing kernel')
    config = traits.String('b02b0.cnf', argstr='--config=%s', usedefault=True,
                           desc=('Name of config file specifying command line '
                                 'arguments'))
    max_iter = traits.String(argstr='--miter=%s',
                          desc='max # of non-linear iterations')
    reg_lambda = traits.String(argstr='--lambda=%s',
                              desc=('lambda weighting value of the '
                                    'regularisation term'))
    ssqlambda = traits.Enum(1, 0, argstr='--ssqlambda=%d',
                            desc=('Weight lambda by the current value of the '
                                  'ssd. If used (=1), the effective weight of '
                                  'regularisation term becomes higher for the '
                                  'initial iterations, therefore initial steps'
                                  ' are a little smoother than they would '
                                  'without weighting. This reduces the '
                                  'risk of finding a local minimum.'))
    regmod = traits.Enum('bending_energy', 'membrane_energy',
                         argstr='--regmod=%s',
                         desc=('Regularisation term implementation. Defaults '
                               'to bending_energy. Note that the two functions'
                               ' have vastly different scales. The membrane '
                               'energy is based on the first derivatives and '
                               'the bending energy on the second derivatives. '
                               'The second derivatives will typically be much '
                               'smaller than the first derivatives, so input '
                               'lambda will have to be larger for '
                               'bending_energy to yield approximately the same'
                               ' level of regularisation.'))
    estmov = traits.String(argstr='--estmov=%s',
                         desc='estimate movements if set')
    minmet = traits.String(argstr='--minmet=%s',
                         desc=('Minimisation method 0=Levenberg-Marquardt, '
                               '1=Scaled Conjugate Gradient'))
    splineorder = traits.Int(3, argstr='--splineorder=%d',
                             desc=('order of spline, 2->Qadratic spline, '
                                   '3->Cubic spline'))
    numprec = traits.Enum('double', 'float', argstr='--numprec=%s',
                          desc=('Precision for representing Hessian, double '
                                'or float.'))
    interp = traits.Enum('spline', 'linear', argstr='--interp=%s',
                         desc='Image interpolation model, linear or spline.')
    scale = traits.Enum(0, 1, argstr='--scale=%d',
                        desc=('If set (=1), the images are individually scaled'
                              ' to a common mean'))
    regrid = traits.Enum(1, 0, argstr='--regrid=%d',
                         desc=('If set (=1), the calculations are done in a '
                               'different grid'))


class MTOPUPOutputSpec(TraitedSpec):
    out_fieldcoef = File(exists=True, desc='file containing the field coefficients')
    out_movpar = File(exists=True, desc='movpar.txt output file')
    out_enc_file = File(desc='encoding directions file output for applytopup')
    out_field = File(desc='name of image file with field (Hz)')
    out_warps = traits.List(File(exists=True), desc='warpfield images')
    out_jacs = traits.List(File(exists=True), desc='Jacobian images')
    out_corrected = File(desc='name of 4D image file with unwarped images')
    out_logfile = File(desc='name of log-file')


class MTOPUP(FSLCommand):
    """
    Interface for FSL topup, a tool for estimating and correcting
    susceptibility induced distortions. See FSL documentation for
    `reference <http://fsl.fmrib.ox.ac.uk/fsl/fslwiki/TOPUP>`_,
    `usage examples
    <http://fsl.fmrib.ox.ac.uk/fsl/fslwiki/topup/ExampleTopupFollowedByApplytopup>`_,
    and `exemplary config files
    <https://github.com/ahheckel/FSL-scripts/blob/master/rsc/fsl/fsl4/topup/b02b0.cnf>`_.

    Examples
    --------

    >>> from nipype.interfaces.fsl import TOPUP
    >>> topup = TOPUP()
    >>> topup.inputs.in_file = "b0_b0rev.nii"
    >>> topup.inputs.encoding_file = "topup_encoding.txt"
    >>> topup.inputs.output_type = "NIFTI_GZ"
    >>> topup.cmdline # doctest: +ELLIPSIS +ALLOW_UNICODE
    'topup --config=b02b0.cnf --datain=topup_encoding.txt \
--imain=b0_b0rev.nii --out=b0_b0rev_base --iout=b0_b0rev_corrected.nii.gz \
--fout=b0_b0rev_field.nii.gz --jacout=jac --logout=b0_b0rev_topup.log \
--dfout=warpfield'
    >>> res = topup.run() # doctest: +SKIP

    """
    _cmd = 'topup'
    input_spec = MTOPUPInputSpec
    output_spec = MTOPUPOutputSpec

    def _format_arg(self, name, trait_spec, value):
        if name == 'encoding_direction':
            return trait_spec.argstr % self._generate_encfile()
        if name == 'out_base':
            path, name, ext = split_filename(value)
            if path != '':
                if not os.path.exists(path):
                    raise ValueError('out_base path must exist if provided')
        return super(MTOPUP, self)._format_arg(name, trait_spec, value)

    def _list_outputs(self):
        outputs = super(MTOPUP, self)._list_outputs()
        del outputs['out_base']
        base_path = None
        if isdefined(self.inputs.out_base):
            base_path, base, _ = split_filename(self.inputs.out_base)
            if base_path == '':
                base_path = None
        else:
            base = split_filename(self.inputs.in_file)[1] + '_base'
        outputs['out_fieldcoef'] = self._gen_fname(base, suffix='_fieldcoef',
                                                   cwd=base_path)
        outputs['out_movpar'] = self._gen_fname(base, suffix='_movpar',
                                                ext='.txt', cwd=base_path)

        n_vols = nb.load(self.inputs.in_file).shape[-1]
        ext = Info.output_type_to_ext(self.inputs.output_type)
        fmt = os.path.abspath('{prefix}_{i:02d}{ext}').format
        outputs['out_warps'] = [
            fmt(prefix=self.inputs.out_warp_prefix, i=i, ext=ext)
            for i in range(1, n_vols + 1)]
        outputs['out_jacs'] = [
            fmt(prefix=self.inputs.out_jac_prefix, i=i, ext=ext)
            for i in range(1, n_vols + 1)]

        if isdefined(self.inputs.encoding_direction):
            outputs['out_enc_file'] = self._get_encfilename()
        return outputs

    def _get_encfilename(self):
        out_file = os.path.join(os.getcwd(),
                                ('%s_encfile.txt' %
                                 split_filename(self.inputs.in_file)[1]))
        return out_file

    def _generate_encfile(self):
        """Generate a topup compatible encoding file based on given directions
        """
        out_file = self._get_encfilename()
        durations = self.inputs.readout_times
        if len(self.inputs.encoding_direction) != len(durations):
            if len(self.inputs.readout_times) != 1:
                raise ValueError(('Readout time must be a float or match the'
                                  'length of encoding directions'))
            durations = durations * len(self.inputs.encoding_direction)

        lines = []
        for idx, encdir in enumerate(self.inputs.encoding_direction):
            direction = 1.0
            if encdir.endswith('-'):
                direction = -1.0
            line = [float(val[0] == encdir[0]) * direction
                    for val in ['x', 'y', 'z']] + [durations[idx]]
            lines.append(line)
        np.savetxt(out_file, np.array(lines), fmt=b'%d %d %d %.8f')
        return out_file

    def _overload_extension(self, value, name=None):
        if name == 'out_base':
            return value
        return super(MTOPUP, self)._overload_extension(value, name)
