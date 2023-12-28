# ep3d-highres-preprocessing
Complex-valued preprocessing pipelines for high-resolution 3D-EPI imaging (e.g. QSM, R2* mapping, MPM)

## basic-preprocessing
Basic nipype preprocessing as well as individual python scripts for separate steps of the preprocessing.

## extended-preprocessing
Extended nipype preprocessing pipeline as developed and used (similarly) in the Rhineland Study (https://www.rheinland-studie.de/en/). One of the differences to the Rhineland Study pipeline is the frequency-based phase matching for multi-TE data. Another difference is compatibility to single-TE data and data acquired using only a single phase encode polarity.