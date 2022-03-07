#!/bin/sh

git pull
cd libLogHelper/
git checkout main && git pull
cd ..
#git submodule update --init --recursive
git add libLogHelper
git commit -m "updating submodule to latest"
git push -u origin master
