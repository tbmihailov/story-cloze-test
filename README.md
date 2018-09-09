# Story-cloze-test models
This is a code for the paper Mihaylov and Frank 2016: [Story Cloze Ending Selection Baselines and Data Examination](http://aclweb.org/anthology/W/W17/W17-0913.pdf)

```bib
@inproceedings{mihaylovfrank:2017,
  author = {Todor Mihaylov and Anette Frank},
  title = {{Story Cloze Ending Selection Baselines and Data Examination}},
  year = {2017},
  booktitle = {Proceedings of the Linking Models of Lexical, Sentential and Discourse-level Semantics – Shared Task},
  address = {Valencia, Spain},
  url = {http://aclweb.org/anthology/W/W17/W17-0913.pdf}
}
```


## Setup environment

### Create virtual environment

```bash
virtualenv venv
```

Activate the environment:
```bash
cd venv
source bin/activate
```


### Install TensorFlow 0.12 with CPU or GPU (CUDA) in virtualenv
Activate the environment
```bash

# Activate the environment
sudo pip install --upgrade virtualenv

# Ubuntu/Linux 64-bit, CPU only, Python 2.7
export TF_BINARY_URL=https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-0.12.1-cp27-none-linux_x86_64.whl

# Mac OS X, CPU only, Python 2.7:
export TF_BINARY_URL=https://storage.googleapis.com/tensorflow/mac/cpu/tensorflow-0.12.1-py2-none-any.whl


# Set CUDA global variables
export LD_LIBRARY_PATH=/usr/local/cuda-7.5/targets/x86_64-linux/lib/:$LD_LIBRARY_PATH
export PATH=/usr/local/cuda-7.5/bin/:$PATH

# Ubuntu/Linux 64-bit, GPU enabled, Python 2.7
# Requires CUDA toolkit 8.0 and CuDNN v5. For other versions, see "Installing from sources" below.
export TF_BINARY_URL=https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-0.12.1-cp27-none-linux_x86_64.whl


# Install tensorflow
pip install --upgrade $TF_BINARY_URL

# test if tensorflow works
python
import TensorFlow as tf # If this does not fail you are okay!
```

### GPU Install TensorFlow with GPU in virtualenv
Activate the environment
```bash
# Login to cluster with GPU units
# HD ICL - https://wiki.cl.uni-heidelberg.de/foswiki/bin/view/Main/FaQ/Tutorials/GridengineTutorial#Quickstart
ssh cluster
# login to the GPU server gpu3 - GTX 1080, 8GB
qlogin -l has_gpu=YES,hostname=gpu03 -q gpu_long.q # get a login on gpu02 in gpu_long.q

# Set CUDA global variables
export LD_LIBRARY_PATH=/usr/local/cuda/targets/x86_64-linux/lib/:$LD_LIBRARY_PATH
export PATH=/usr/local/cuda/bin/:$PATH

# activate the environment
sudo pip install --upgrade virtualenv

# Ubuntu/Linux 64-bit, GPU enabled, Python 2.7
# Requires CUDA toolkit 7.5 and CuDNN v4. For other versions, see "Install from sources" below.
export TF_BINARY_URL=https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-0.12.0rc1-cp27-none-linux_x86_64.whl

# Install tensorflow
pip install --upgrade $TF_BINARY_URL

# test if tensorflow works
python
import TensorFlow as tf # If this does not fail you are okay!
```

### Install everything from requirements.txt
pip install -r requirements.txt

### Install Jupyter (for experiments)
sudo pip install jupyter

### Download CoreNLP
\corenlp\download_core_nlp.sh

### Install PyWrapper https://github.com/brendano/stanford_corenlp_pywrapper
git clone https://github.com/brendano/stanford_corenlp_pywrapper
cd stanford_corenlp_pywrapper
pip install .

### Install Java 8 (required for Stanford CoreNLP)
https://www.digitalocean.com/community/tutorials/how-to-install-java-with-apt-get-on-ubuntu-16-04

sudo apt-get update
sudo add-apt-repository ppa:webupd8team/java
sudo apt-get update
sudo apt-get install oracle-java8-installer
sudo update-alternatives --config java
sudo nano /etc/environment
source /etc/environment

### Word Sense Disambigaution (if used when parsing)
```
pip install nltk
python -m nltk.downloader 'punkt'

python -m nltk.downloader 'averaged_perceptron_tagger'
pip install pywsd
```

## Data and embeddings
The training data is placed in resources/roc_stories_data

Debug experiments are set to run with a small 20d embeddings in `resources/word2vec/`
To run full experiments, download word embeddings from:
* Google 300d embeddings: https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit?usp=sharing
* Numberbatch embeddings: https://github.com/commonsense/conceptnet-numberbatch
Update the paths in the `_server.sh` scripts to match your embeddings local path.

Experiments with logistic regression features baseline
```
# debug (uses small 20d embeddings)
bash scripts/exp/story_cloze_end/story_cloze_v1_baseline_sim_v1_run.sh

# full experiments
bash scripts/exp/story_cloze_end/story_cloze_v1_baseline_sim_v1_run_server.sh

```

Experiments with plain LSTM neural baseline with Attention
```
# debug (uses small 20d embeddings)
bash scripts/exp/story_cloze_end/story_cloze_v4_lstm_att_run.sh

# full experiments
bash scripts/exp/story_cloze_end/story_cloze_v4_lstm_att_run_server.sh
```

Experiments with memory networks
```
# debug (uses small 20d embeddings)
bash scripts/exp/story_cloze_end/story_cloze_v5_mem_run.sh

# full experiments (requires
bash scripts/exp/story_cloze_end/story_cloze_v5_mem_run_server.sh
```