#!/bin/bash

if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <kernel name> <num rows>"
    exit -1
fi

kernel_name=$1
num_rows=$2

cd CUDA

cp detect_main_c.template detect_main.c
sed -i 's/PIM_KERNEL_INSTANTIATION/pim_state_t *pim_state = init_pim('"${kernel_name}"', 1048576, '"${num_rows}"');/g' detect_main.c

make OUTPUT=1
#rm detect_main.c

mkdir -p ../bin
mv leukocyte ../bin/leukocyte_${kernel_name}_${num_rows}
