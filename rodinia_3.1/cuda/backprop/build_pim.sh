#!/bin/bash

if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <kernel name> <num rows>"
    exit -1
fi

kernel_name=$1
num_rows=$2

cp backprop_cuda_cu.template backprop_cuda.cu
sed -i 's/PIM_KERNEL_INSTANTIATION/pim_state_t *pim_state = init_pim('"${kernel_name}"', 1048576, '"${num_rows}"');/g' backprop_cuda.cu

make
rm backprop_cuda.cu

mkdir -p bin
mv backprop bin/backprop_${kernel_name}_${num_rows}
