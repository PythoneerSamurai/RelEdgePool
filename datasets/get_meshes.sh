#!/usr/bin/env bash
wget 'https://www.dropbox.com/scl/fi/fl743u9mqcxp8n4vv77e4/meshes.tar.gz?rlkey=l2w9n264baqysxfpjx1b695bx&st=ccwpm3dw&dl=0' -O .meshes.tar.gz
tar -xzvf .meshes.tar.gz && rm .meshes.tar.gz
echo "downloaded and extracted data"
