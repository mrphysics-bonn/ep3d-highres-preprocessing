"""
#Rhineland Study MRI Post-processing pipelines
#rs_qsm_pipeline: QSMEPI data preprocessing using FSL/ANTs/Matlab and python with nipype workflow
"""
import os
import sys
from glob import glob

if os.path.exists('MANIFEST'): os.remove('MANIFEST')

def main(**extra_args):
    from setuptools import setup
    setup(name='qsm_pipeline',
        version='1.0.0',
        description='collection of routines to process QSM data',
        long_description="""RhinelandStudy processing for QSM EPI scans.""",
        author='Ruediger Stirnberg',
        author_email='ruediger.stirnberg@dzne.de',
        license='MIT',
        packages= ['qsm_pipeline'
                   ],
        package_data = { },
        include_package_data=True,
        entry_points={ 'console_scripts': ['run_qsm_pipeline=qsm_pipeline.run_qsm_pipeline:main'
                                          ]},
        classifiers = [c.strip() for c in """\
            Development Status :: 1 - Beta
            Intended Audience :: Developers
            Intended Audience :: Science/Research
            Operating System :: OS Independent
            Programming Language :: Python
            Topic :: Software Development
            """.splitlines() if len(c.split()) > 0],
          maintainer = 'RheinlandStudy MRI/MRI-IT group, DZNE',
          maintainer_email = 'mohammad.shahid@dzne.de',
          install_requires=['nibabel', 'nipype', 'nipy', 'numpy', 'scipy','matplotlib','pycrypto','psycopg2','pyxnat'],
          zip_safe=False,
          **extra_args
          )

if __name__ == "__main__":
    main()
