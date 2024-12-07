# OPCR

## Datasets

We consider three sets of datasets in this paper.
- **Traffic dataset**: The python library [torch-spatiotemporal](https://torch-spatiotemporal.readthedocs.io/en/latest/) provides two preprocessed datasets (PEMS-BAY, METR-LA).

- **Large-scale dataset**: The downloaded datasets are suggested to be stored in `./data/pvus` and `./data/cer`, respectively. `./data/pre_pvus.py` and `./data/pre_cer.py` are used to preprocess these two datasets.
    - **PV-US**: This dataset can be downloaded from https://www.nrel.gov/grid/solar-power-data.html.
    - **CER-E**: This dataset can be downloaded from https://www.ucd.ie/issda/data/commissionforenergyregulationcer/.

- **Traffic4cast dataset**: This dataset can be downloaded from this [project](https://github.com/iarai/NeurIPS2022-traffic4cast). 

## Requirement
See `requirements.txt` for the list of packages.

## Experiments

- **Traffic dataset**: `main_traffic.py` is used to train the proposed OPCR on traffic dataset.
```py
"""
gpu: device id
dataset-name: "bay" or "la"
missing-type: "point" or "spatial"
missing-rate: [0.0, 1.0]
seed: random seed
"""
python main_traffic.py --gpu 0 --dataset-name bay --missing-type point --missing-rate 0.95 --seed 0
```

- **Large-scale dataset**: `main_large.py` is used to train the proposed OPCR on large-scale dataset.
```py
"""
gpu: device id
dataset-name: "pvus" or "cer"
missing-type: "point" or "spatial"
missing-rate: [0.0, 1.0]
seed: random seed
"""
python main_large.py --gpu 0 --dataset-name pvus --missing-type point --missing-rate 0.95 --seed 0
```

- **Traffic4cast dataset**: `main_t4c_con.py` and `main_t4c_seg.py` are used to train the proposed OPCR on t4c dataset.
```py
"""
gpu: device id
city: "london" or "madrid" or "melbourne"
"""
# congestion classification task
python main_t4c_con.py --gpu 0 --city london
# travel time prediction task
python main_t4c_seg.py --gpu 0 --city london
```

## Citation
```bibtex
@inproceedings{denglearning,
  title={Learning from Highly Sparse Spatio-temporal Data},
  author={Deng, Leyan and Wu, Chenwang and Lian, Defu and Chen, Enhong},
  booktitle={The Thirty-eighth Annual Conference on Neural Information Processing Systems}
}
```

## Acknowledgement
Our experiments on the traffic dataset and strategy for injecting missing data are based on the implementations of [SPIN](https://github.com/Graph-Machine-Learning-Group/spin/tree/main?tab=readme-ov-file). 
We gratefully thanks for their contribution.
