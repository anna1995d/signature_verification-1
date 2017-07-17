#!/usr/bin/env bash

rsync -a --ignore-existing configuration.sample.yaml configuration.yaml
