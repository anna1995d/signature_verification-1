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
pip install /tmp/tensorflow_pkg/<tensorflow_package> 
```

### Configuration
The configuraion must be in _*`configuration.json`*_. For a sample configuration see the provided sample [here][sample-config]. 

#### Logger
_Configuration for logs during training phase._
##### Log format
Logger output format.
##### Log file
Logger output file.
##### Log level
Availble choices are _**`critical`**_, _**`fatal`**_, _**`error`**_, _**`warning`**_, _**`info`**_, _**`debug`**_ and _**`notset`**_.

#### Export
_Configuration for exporting models during training phase._
##### Model save template
Template for saving models after training phase, it must have _**`name`**_ attribute.

#### Data
_Configuration for reading data during training phase._
##### User count
Count of different users.
##### Input dimension
Count of input data features. 
##### Genuine sample count
Count of genuine samples for each user.
##### Forged sample count
Count of forged samples for each user.
##### Genuine path template
Template for genuine data, it must have _**`user`**_ and _**`sample`**_ attributes.
##### Forged path template
Template for forged data, it must have _**`user`**_, _**`sample`**_ and _**`forger`**_ attributes.

#### Autoencoder
_Configuration for training phase of Autoencoder._
##### Batch size
Count of samples to be processed at each step of training.
##### Encoded length
###### Start
Least length for encoded representation.
###### Finish
Most length for encoded representation.
###### Step
Increasing length for encoded representation by this value.
##### Train epochs
###### Start
Least count of epochs for training phase.
###### Finish
Most count of epochs for training phase.
###### Step
Increase count of epochs for training phase by this value.
##### Cell types
Availble choices are _**`LSTM`**_, _**`GRU`**_ and _**`SimpleRNN`**_.

##### Sample configuration
If you want to use a pre-written configuration, you could use the provided sample _**`configuration.sample.json`**_.  
```bash
rsync -a --ignore-existing configuration.sample.json configuration.json
```

[tf-doc]: https://www.tensorflow.org/install/install_sources/
[anaconda-doc]: https://docs.continuum.io/anaconda/install/
[virtualenv-doc]: https://virtualenv.pypa.io/en/stable/installation/
[sample-config]: https://github.com/kahrabian/signature_verification/blob/master/configuration.sample.json