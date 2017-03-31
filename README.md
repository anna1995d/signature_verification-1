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

[tf-doc]: https://www.tensorflow.org/install/install_sources/
[anaconda-doc]: https://docs.continuum.io/anaconda/install/
[virtualenv-doc]: https://virtualenv.pypa.io/en/stable/installation/
