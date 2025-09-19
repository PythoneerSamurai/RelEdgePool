#!/usr/bin/env bash
wget 'https://www.dropbox.com/scl/fi/mpswmpjhditajscny5z0r/cubes.tar.gz?rlkey=5v43nf7k1hv5oqu8d1trw81ye&st=epls0t7k&dl=1' -O .cubes.tar.gz
tar -xzvf .cubes.tar.gz && rm .cubes.tar.gz
echo "Downloaded and extracted data"

