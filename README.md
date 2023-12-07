# Enhancing Representation Power of Deep Neural Networks With Negligible Parameter Growth for Industrial Applications

This repository contains the official implementation of the paper "Enhancing Representation Power of Deep Neural Networks With Negligible Parameter Growth for Industrial Applications." 

## Usage

### Defect detection
Download the [KolektorSDD2 data](https://www.vicos.si/Downloads/KolektorSDD2).

```bash
cd defect_detection
nohup ./EXPERIMENTS_COMIND.sh  > log.txt 2>&1 &
```

### Image classification
```bash
cd image_classification
nohup python3 GPUsRuns.py  > log.txt 2>&1 &
```

### Critical temperature prediction of superconductors

Download the [Superconductivty data](https://archive.ics.uci.edu/dataset/464/superconductivty+data).
```bash
cd superconductors
```
Run cells in superconductors.ipynb.

## Acknowledgements

The defect detection code is based on the [Mixed-SegDec-Net](https://github.com/vicoslab/mixed-segdec-net-comind2021) project. We modified the orignal model architecture.
