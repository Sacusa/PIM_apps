#!/bin/bash

if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <kernel name> <num rows>"
    exit -1
fi

kernel_name=$1
num_rows=$2

cp hotspot_cu.template hotspot.cu
sed -i 's/PIM_KERNEL_INSTANTIATION/pim_state_t *pim_state = init_pim('"${kernel_name}"', 1048576, '"${num_rows}"');/g' hotspot.cu

make
rm hotspot.cu

mkdir -p bin
mv hotspot bin/hotspot_${kernel_name}_${num_rows}
