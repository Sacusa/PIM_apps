#!/bin/bash

if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <kernel name> <num rows>"
    exit -1
fi

kernel_name=$1
num_rows=$2

cp main_c.template main.c
sed -i 's/PIM_KERNEL_INSTANTIATION/pim_state_t *pim_state = init_pim('"${kernel_name}"', 1048576, '"${num_rows}"');/g' main.c

make
rm main.c

mkdir -p bin
mv b+tree.out bin/b+tree_${kernel_name}_${num_rows}
