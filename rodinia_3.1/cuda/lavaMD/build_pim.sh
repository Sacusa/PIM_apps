#!/bin/bash

if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <kernel name> <num rows>"
    exit -1
fi

kernel_name=$1
num_rows=$2

cd kernel

cp kernel_gpu_cuda_wrapper_cu.template kernel_gpu_cuda_wrapper.cu
sed -i 's/PIM_KERNEL_INSTANTIATION/pim_state_t *pim_state = init_pim('"${kernel_name}"', 1048576, '"${num_rows}"');/g' kernel_gpu_cuda_wrapper.cu

cd ..
make
mkdir -p bin
mv lavaMD bin/lavaMD_${kernel_name}_${num_rows}

cd kernel
rm kernel_gpu_cuda_wrapper.cu
