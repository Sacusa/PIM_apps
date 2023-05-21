#!/bin/bash

if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <kernel name> <num rows>"
    exit -1
fi

kernel_name=$1
num_rows=$2

cp streamcluster_cuda_cpu_cpp.template streamcluster_cuda_cpu.cpp
sed -i 's/PIM_KERNEL_INSTANTIATION/pim_state_t *pim_state = init_pim('"${kernel_name}"', 1048576, '"${num_rows}"');/g' streamcluster_cuda_cpu.cpp

make
rm streamcluster_cuda_cpu.cpp

mkdir -p bin
mv sc_gpu bin/streamcluster_${kernel_name}_${num_rows}
