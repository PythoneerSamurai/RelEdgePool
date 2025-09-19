#!/usr/bin/env bash
wget 'https://www.dropbox.com/scl/fi/4ppp8o09mqqvugd74msxw/shrec.tar.gz?rlkey=ddxkmmsmsdi1rsobiohw5d19q&st=zpgevqjb&dl=0' -O .shrec.tar.gz
tar -xzvf .shrec.tar.gz && rm .shrec.tar.gz
echo "downloaded and extracted data"
