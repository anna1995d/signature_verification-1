Signature Verification
===

### Setup Environment
In order to prevent package conflicts for different projects, it is recommended to install a python environment manager 
such as `virualenv`. Because of its rich features I personally use `anaconda` which provides similar functionallity to 
`virtualenv` with more features such as creating environment with different python versions without actually installing 
it globally on system. For more information on installing `anaconda` see the [documentations][anaconda-doc]. (Since 
`anaconda` has many preinstalled packages which might not be needed, you could install the minimal version, `miniconda`)

If you prefer to use `virtualenv` you could find the installation instruction in [documentations][virtualenv-doc].

```bash
conda create -n sigv python=3  # Create a python 3 environment with the name of sigv
```

### Install requirements

```bash
source activate sigv  # Activate project environment
conda install ipython jupyter matplotlib numpy h5py  # Install packages available on conda repository
pip install keras  # Insallt packages available on pypi repository
```

### Install tensorflow from source
It is possible to install both CPU and GPU versions of `tensorflow` via `pip` but because of the optimizations and 
customizations which are available, it is recommended to install `tensorflow` from source. For more information on 
configuring `tensorflow` installation see the [documentations][tf-doc].

```bash
git submodule update --init
cd libs/tensorflow
./configure
bazel build --config=opt //tensorflow/tools/pip_package:build_pip_package
bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tensorflow_pkg
pip install /tmp/tensorflow_pkg/[tensorflow_package] 
```

### Configuration
The configuraion must be in _`configuration.json`_. For a sample configuration see the provided sample [here][sample-config]. 

#### Logger
_Configuration for logs during training phase._

Key | Value | Description
:---: | :---: | :---:
`log_format` | `str`, available attributes are `%(asctime)s`, `%(name)s`, `%(levelname)s` and `%(message)s`  | logger output format
`log_file` | `str` | logger output file path
`log_level` | `str`, available choices are `critical`, `fatal`, `error`, `warning`, `info`, `debug` and `notset` | logger logging level

#### Export
_Configuration for exporting models during training phase._

Key | Value | Description
:---: | :---: | :---:
`model_save_template` | `str`, it must contain `{name}` | path template for saving models

#### Data
_Configuration for reading data during training phase._

Key | Value | Description
:---: | :---: | :---:
`user_count` | `int` | count of different users
`input_dimension` | `int` | count of input data features
`genuine_sample_count` | `int` | count of genuine samples for each user
`forged_sample_count` | `int` | count of forged samples for each user
`forger_count` | `int` | count of forgers per user
`genuine_path_template` | `str`, it must contain `{user}` and `{sample}` | path template for genuine data
`forged_path_template` | `str`, it must contain `{user}`, `{sample}` and `{forger}` | path template for forged data

#### Autoencoder
_Configuration for training phase of Autoencoder._

Key | Value | Description
:---: | :---: | :---:
`batch_size` | `int` | count of samples to be processed at each step of training
`encoded_length` | `json`, it must contain `start`, `finish` and `step` | length of encoded representaion
`train_epochs` | `json`, it must contain `start`, `finish` and `step` | count of epochs during training phase
`cell_types` | `list`, availble choices are `LSTM`, `GRU` and `SimpleRNN` | list of cells to use for training phase

#### Sample configuration
If you want to use a pre-written configuration, you could use the provided sample _`configuration.sample.json`_.  

```bash
rsync -a --ignore-existing configuration.sample.json configuration.json
```

[tf-doc]: https://www.tensorflow.org/install/install_sources/
[anaconda-doc]: https://docs.continuum.io/anaconda/install/
[virtualenv-doc]: https://virtualenv.pypa.io/en/stable/installation/
[sample-config]: https://github.com/kahrabian/signature_verification/blob/master/configuration.sample.json