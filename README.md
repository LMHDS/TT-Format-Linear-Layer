默认配置 (4个核心):
mkdir build && cd build
cmake ..
make


自定义核心数量:
cmake -DTT_NUM_CORES=3 ..


自定义模式：
cmake -DTT_NUM_CORES=3 \
      -DTT_USE_CUSTOM_MODES=ON \
      -DTT_INPUT_MODES="4,4,4" \
      -DTT_OUTPUT_MODES="4,4,4" \
      -DTT_RANKS="1,64,64,1" ..


