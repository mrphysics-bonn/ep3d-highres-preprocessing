# -*- coding: utf-8 -*-
"""A nipype pre-processing pipeline for complex-valued, 3D-EPI-based R2*/QSM data as used in the rhineland study (RLS).

Created on Fri Jun 7 2019

@authors: stirnbergr, shahidm
@email: ruediger.stirnberg@dzne.de
"""

from __future__ import division
import textwrap
import argparse
import sys
import os
import glob
from itertools import chain
from nipype import config, logging
import numpy as np

from .QsmEpiProcessing import create_qsmepiwf

import matplotlib
matplotlib.use('Agg')


def main():
    """
    Command line wrapper for preprocessing QSM-EPI data
    """
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter,
                                     description='Prepocessing pipeling for complex-valued, 3D-EPI-based R2*/QSM data as used in the rhineland study (RLS).',
                                     epilog=textwrap.dedent('''\
                                     Example: {prog} --scansdir subjects/ --subjects 01 02
                                     -o output -w work -p 2 -c 6 -m 2

                                     Input data is expected as a 4D Nifti dataset per
                                     phase encode direction and magnitude and phase:
                                     *_AP.nii.gz, *_AP_Phase.nii.gz, *_PA.nii.gz, *_PA_Phase.nii.gz

                                     Echoes (and averages) are expected in chronological scanning order.
                                     The RLS protocol, for example:
                                     multi-echo-shots=2, multi-echoes=6 (3 rephased TE per shot),
                                     2 measurements per PE direction ==>
                                     TE 1, 3, 5, 2, 4, 6, 1, 3, 5, 2, 4, 6

                                     In the RLS, DICOMs were originally saved without mosaic option,
                                     and dcmstack is used for Nifti conversion using options
                                     --extract-private --embed-meta to keep complete header information.
                                     This immediately results in the expected chronological order.

                                     If data are stored with the newer mosaic option to save space,
                                     a Dicom series is created per echo time (E00...) and magnitude (M)
                                     and phase (P), e.g for RLS:
                                     E00_M/P: TE 1, 1; E01_M/P: TE 3, 3; E02_M/P: TE 5, 5;
                                     E03_M/P: TE 2, 2; E04_M/P: TE 4, 4; E05_M/P: TE 6, 6;
                                     Therefore, either before or after Nifti conversion, currently the
                                     data has to be brought into the right shape before calling {prog},
                                     e.g. by combining fslroi and fslmerge.

                                            ''').format(prog=os.path.basename(sys.argv[0])))
                                     #formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('-s', '--scansdir', help='Scans directory where scans'\
                        ' for each subjects are located, e.g.: subjects/',required=True)

    parser.add_argument('--subjects', help='One or more scansdir-subdirectory names (space separated), e.g 01 02. '
                                           'Each must contain two *AP*.nii.gz and two *PA*.nii.gz files (unless '
                                           'otherwise specified by --blip-up-down option). subdirectory names not '
                                           'specified, all subdirectories of datadir are treated as subject IDs.',
                        default=None, required=False, nargs='+', action='append')
    parser.add_argument('-o', '--outputdir', help='Output directory', required=True)
    parser.add_argument('-w', '--workdir', help='working directory', required=True)
    parser.add_argument('-d', '--debug', help='debug mode', action='store_true')
    parser.add_argument('-p', '--processes', help='parallel processes', default=1, type=int)
    parser.add_argument('-t', '--threads', help='ITK threads', default=1, type=int)
    parser.add_argument('-c', '--contrasts', help='Total number contrasts in protocol (e.g. different TEs in QSM, or TEs x multi-contrasts in MPM)', required=True, type=int)
    parser.add_argument('-m', '--multi-contrasts', help='Number of different base contrasts (e.g. Multi-echo shots set for multi-echo segmentation, or multi-contrasts in MPM: '
                                                         'contrasts = multi-contrasts * rephased-multi-echoes. '
                                                         '', required=True, type=int)
    parser.add_argument('-e', '--echo-times', help='Specify actual echo times [ms] in the following order: '
                                                    'All rephased echo times of first multi-contrast '
                                                    'followed by the first echo times of subcequent multi-contrasts. '
                                                    'A complete TE matrix is computed internally based on these values. '
                                                    'A better, informed phase correction can then be performed, in a '
                                                    'single step instead of pseudo freq. correction followed by additional '
                                                    'phase matching. Also, the stored frequency change matrix is in [Hz].'
                                                    'NOTE: '
                                                    'just like the entire pipeline, this assumes that the rephased echo time '
                                                    'spacings are the same across all multi-contrasts.'
                                                    '', required=False, default=None, type=float, nargs='+')
    parser.add_argument('-f', '--phase-factor', help='Factor multiplied to the phase data to convert to radian. '
                                                     'If phase data is already in radians, enter 1.0. '
                                                     'Default assumes that phase is scaled from -4096 to 4096.', default=np.pi/4096.0, type=float)

    parser.add_argument('-b', '--blip-up-down', help='Blip up/down mode for distortion correction, or not. '
                                                     '0: both AP and PA; '
                                                     '1: only AP; '
                                                     '2: only PA. '
                                                     'In case of 1 or 2, distortion estimation (topup) and correction '
                                                     '(applytopup) steps are skipped and the final k-space combination '
                                                     'is a simple complex average across measurements.',
                                                     required=False, type=int, default=0, choices=[0,1,2])

    parser.add_argument('-a', '--averaging', help=   'Averaging order. '
                                                     '0: long, repetition of the same TE/contrast is (0,N/2) and so on; '
                                                     '1: short; repetition of the same TE/contrast is (0,1) and so on;',
                                                     required=False, type=int, default=0, choices=[0,1])

    args = parser.parse_args()

    nthreads         = args.threads
    phase_factor     = args.phase_factor
    multi_echoes     = args.contrasts
    multi_echo_shots = args.multi_contrasts
    echo_times       = args.echo_times
    blip_up_down     = args.blip_up_down
    av_order         = args.averaging

    print("av_order: %i"  %av_order)

    # Basic error handling
    if not np.mod(multi_echoes, multi_echo_shots) == 0:
        raise AssertionError('The entered number of multi-echoe-shots %d must be a factor of the multi-echoes %d' % (multi_echo_shots, multi_echoes))

    rephased_echoes = multi_echoes//multi_echo_shots
    if echo_times is not None and len(echo_times) != rephased_echoes + (multi_echo_shots-1):
        raise AssertionError('%d echo times should be entered: %d rephased TEs of the first contrast (multi-echo-shot), followed by %d first TEs of the subsequent contrasts.' % (rephased_echoes + (multi_echo_shots-1), rephased_echoes, multi_echo_shots-1))

    # Create the workflow
    scans_dir  = os.path.abspath(os.path.expandvars(args.scansdir))
    if not os.path.exists(scans_dir):
        raise IOError("Scans directory does not exist.")

    subject_ids = []

    if args.subjects:
        subject_ids = list(chain.from_iterable(args.subjects))
    else:
        subject_ids = glob.glob(scans_dir.rstrip('/') + '/*')
        subject_ids = [os.path.basename(s.rstrip('/')) for s in subject_ids]



    if not os.path.exists(args.workdir):
        os.makedirs(args.workdir)

    if not os.path.exists(args.outputdir):
        os.makedirs(args.outputdir)


    config.update_config({
        'logging': {'log_directory': args.workdir, 'log_to_file': False},
        'execution': {'job_finished_timeout' : 60,
                      'poll_sleep_duration' : 30,
                      'hash_method' : 'content',
                      'local_hash_check' : False,
                      'stop_on_first_crash':True,
                      'crashdump_dir': args.workdir,
                      'crashfile_format': 'txt'#,
                      #'remove_unnecessary_outputs':False
                       },
                       })

    #config.enable_debug_mode()
    logging.update_logging(config)

    qsmepiwf = create_qsmepiwf(data_dir=scans_dir, subj_ids=subject_ids, phase_factor=phase_factor, multi_echoes=multi_echoes, multi_echo_shots=multi_echo_shots, nthreads=nthreads, blip_up_down=blip_up_down, av_order=av_order, echo_times=echo_times)

    qsmepiwf.base_dir = args.workdir
    qsmepiwf.inputs.inputnode.outputdir = os.path.abspath(os.path.expandvars(args.outputdir))

    # Visualize workflow
    if args.debug:
        qsmepiwf.write_graph(graph2use='colored', simple_form=True)


    # Run workflow
    qsmepiwf.run(plugin='MultiProc', plugin_args={'n_procs' : args.processes})

    print('DONE!!!')


if __name__ == '__main__':
    sys.exit(main())
