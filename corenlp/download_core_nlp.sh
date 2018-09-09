#!/usr/bin/env bash

wget "http://nlp.stanford.edu/software/stanford-corenlp-full-2015-12-09.zip"  "stanford-corenlp-full-2015-12-09.zip"
# sudo apt-get install unzip
unzip stanford-corenlp-full-2015-12-09.zip 
rm stanford-corenlp-full-2015-12-09.zip

# coref parser
wget http://nlp.stanford.edu/software/stanford-srparser-2014-10-23-models.jar