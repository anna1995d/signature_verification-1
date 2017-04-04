#!/usr/bin/env bash

rsync -a --ignore-existing configuration.sample.json configuration.json
