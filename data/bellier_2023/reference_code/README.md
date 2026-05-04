# PF_HFAdecoding
- This repository contains all the MATLAB and Python code produced and used in our paper "Encoding and decoding analysis of music perception using intracranial EEG" ([see latest version on bioRxiv](https://www.biorxiv.org/content/10.1101/2022.01.27.478085v2)).
- PF_MASTERSCRIPT.m lists all the steps performed (preprocessing, encoding analyses, and decoding analyses).
- The upstream release also ships ``PF_NeMo_MASTERSCRIPT.py`` (Neural Modeling toolbox / NeMo). This course fork omits that file because nothing in ``logic/`` imports it; fetch the original Bellier release if you need the NeMo driver.
- Example parameter files (``params8.txt``, ``params96.txt``) still illustrate the encoding / nonlinear decoding settings from the paper.
