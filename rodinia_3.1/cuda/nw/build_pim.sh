#!/bin/bash

if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <kernel name> <num rows>"
    exit -1
fi

kernel_name=$1
num_rows=$2

cp needle_cu.template needle.cu
sed -i 's/PIM_KERNEL_INSTANTIATION/pim_state_t *pim_state = init_pim('"${kernel_name}"', 1048576, '"${num_rows}"');/g' needle.cu

make
rm needle.cu

mkdir -p bin
mv needle bin/nw_${kernel_name}_${num_rows}
