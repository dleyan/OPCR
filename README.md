# OPCR

## Datasets

We consider three sets of datasets in this paper.
- **Traffic dataset**: The python library [torch-spatiotemporal](https://torch-spatiotemporal.readthedocs.io/en/latest/) provides two preprocessed datasets (PEMS-BAY, METR-LA).

- **Large-scale dataset**: The downloaded datasets are suggested to be stored in `./data/pvus` and `./data/cer`, respectively. `./data/pre_pvus.py` and `./data/pre_cer.py` are used to preprocess these two datasets.
    - **PV-US**: This dataset can be downloaded from https://www.nrel.gov/grid/solar-power-data.html.
    - **CER-E**: This dataset can be downloaded from https://www.ucd.ie/issda/data/commissionforenergyregulationcer/.

- **Traffic4cast dataset**: This dataset can be downloaded from this [project](https://github.com/iarai/NeurIPS2022-traffic4cast). 

## Acknowledgement
Our experiments on the traffic dataset and strategy for injecting missing data are based on the implementations of [SPIN](https://github.com/Graph-Machine-Learning-Group/spin/tree/main?tab=readme-ov-file). 
We gratefully thanks for their contribution.
