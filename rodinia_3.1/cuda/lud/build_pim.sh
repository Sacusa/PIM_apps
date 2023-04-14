#!/bin/bash

if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <kernel name> <num rows>"
    exit -1
fi

kernel_name=$1
num_rows=$2

cd cuda

cp lud_kernel_cu.template lud_kernel.cu
sed -i 's/PIM_KERNEL_INSTANTIATION/pim_state_t *pim_state = init_pim('"${kernel_name}"', 1048576, '"${num_rows}"');/g' lud_kernel.cu

make
rm lud_kernel.cu

mkdir -p ../bin
mv lud_cuda ../bin/lud_${kernel_name}_${num_rows}
