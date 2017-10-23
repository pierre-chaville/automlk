#!/usr/bin/env bash

PATH=$PATH:$HOME/anaconda3/bin
cd $1
make html
make latexpdf