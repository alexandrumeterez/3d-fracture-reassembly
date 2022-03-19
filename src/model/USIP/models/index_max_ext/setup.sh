#!/bin/sh

bsub -R rusage[mem=10000,ngpus_excl_p=1] python3 setup.py install --install-lib /cluster/home/ameterez/.local/lib/python3.8/site-packages