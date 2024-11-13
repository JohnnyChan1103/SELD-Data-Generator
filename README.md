# Spatial Room Impulse Response (SRIR) Generater

This repository is for generating sound event localization and detection (SELD) samples, including microphone-array-format signals and first-order-ambisonics-format signals.

## Requirements

Download FSD50K (sound event samples)): [https://zenodo.org/records/4060432](https://zenodo.org/records/4060432)

Download TAU-SRIR DB (spatial room impulse responses, optional)): [https://zenodo.org/records/6408611](https://zenodo.org/records/6408611)

The Python version is 3.11.8. 

If you need to generate first-order-ambisonic format signals, please download matlab R2023b and [Matlab Engine](https://ww2.mathworks.cn/help/matlab/matlab-engine-for-python.html) for Python 3.11.8. The corresponding scripts are implemented by Matlab.

Run the following command to install necessary Python packages.

```shell
pip install -r requirements.txt
```

# Quick Start

The `database` directory should contain the FSD50K and TAU-SRIR DB datasets:

<pre>
./database
├────FSD50K
│       ├── FSD50K.dev_audio
│       ├── FSD50K.doc
│       ├── FSD50K.eval_audio
│       └── ...
├────TAU-SRIR_DB
│       ├── TAU-SNoise_DB
│       └── TAU-SRIR_DB
└─...
</pre>

Then run:

```shell
python make_dataset.py 1 # synthesize SELD samples using simulated SRIRs
```

or run:

```shell
python make_dataset.py 2 # synthesize SELD samples using collected SRIRs from TAU-SRIR DB
```

## Others

This repository contains several files, which in total create a complete data generation framework.

* The `get_parameters.py` is a separate script used for setting the parameters for the data generation process.
* The `make_dataset.py` is the main script in which the whole framework is used to perform the full data generation process.
* The `utils.py` is an additional file containing complementary functions for other scripts.
* The `data_generator/db_config.py` is a class for containing audio filelists and data parameters from audio datasets used for the mixture generation.
* The `data_generator/data_synthesis.py` is a class for data synthesis using simulated SRIRs.
* The `data_generator/data_synthesis_test.py` is a class for data synthesis using collected SRIRs from TAU-SRIR DB.
* The `srir` contains the scripts of simulating RIRs and first-order-ambisonics converting.
* The `source_datasets/material_absorption` contains material absorption coefficients, and more details can be found in [2].
* The `source_datasets/single_source_samples` contains selected single-source sound event clips from FSD50K, and more details can be found in [2].

Moreover, one object file is included in case that the database configuration via `db_config.py` takes too much time:

* The `db_config_fsd50k.obj` is a DBConfig class containing information about the database and files for the FSD50K dataset.

## Cite

Please consider citing our paper if you find this code useful for your research.

[1] Jinbo Hu, Yin Cao, Ming Wu, Qiuqiang Kong, Feiran Yang, Mark D. Plumbley, and Jun Yang, "Selective-Memory Meta-Learning with Environment Representations for Sound Event Localization and Detection," IEEE/ACM Transactions on Audio, Speech, and Language Processing (TASLP), 2024. [URL](https://arxiv.org/abs/2312.16422)

[2] Jinbo Hu, Yin Cao, Ming Wu, Fang Kang, Feiran Yang, Wenwu Wang, Mark D. Plumbley, Jun Yang, "PSELDNets: Pre-trained Neural Networks on Large-scale Synthetic Datasets for Sound Event Localization and Detection" arXiv:2411.06399, 2024. [URL](https://arxiv.org/abs/2411.06399)

## External Links

1. [https://github.com/danielkrause/DCASE2022-data-generator](https://github.com/danielkrause/DCASE2022-data-generator)
2. [https://github.com/chris-hld/spaudiopy](https://github.com/chris-hld/spaudiopy)
3. [https://github.com/AppliedAcousticsChalmers/ambisonic-encoding]()
