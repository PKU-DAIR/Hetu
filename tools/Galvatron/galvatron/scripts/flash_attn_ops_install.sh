git clone --recursive https://github.com/Dao-AILab/flash-attention.git
pip3 install flash-attention/csrc/fused_dense_lib
pip3 install flash-attention/csrc/layer_norm
pip3 install flash-attention/csrc/rotary
pip3 install flash-attention/csrc/xentropy
rm -rf flash-attention