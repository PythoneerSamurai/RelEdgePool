#!/usr/bin/env bash
wget 'https://www.dropbox.com/scl/fi/jouw9ss5gl0ydip0hbaw1/single_iteration_pooled_meshes.tar.gz?rlkey=nixz3w5h4j4r505f9fbitmbyy&st=qi62uhqa&dl=0' -O .pooled_meshes.tar.gz
tar -xzvf .pooled_meshes.tar.gz && rm .pooled_meshes.tar.gz
echo "downloaded and extracted data"
