#!/bin/bash

if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <kernel name> <num rows>"
    exit -1
fi

pim_kernel=$1
num_rows=$2

declare -a kernels=("euler3d" "euler3d_double" "pre_euler3d" "pre_euler3d_double")

for kernel in "${kernels[@]}"; do
    cp ${kernel}_cu.template ${kernel}.cu
    sed -i 's/PIM_KERNEL_INSTANTIATION/pim_state_t *pim_state = init_pim('"${pim_kernel}"', 1048576, '"${num_rows}"');/g' ${kernel}.cu
done

make
mkdir -p bin

for kernel in "${kernels[@]}"; do
    rm ${kernel}.cu
    mv ${kernel} bin/${kernel}_${pim_kernel}_${num_rows}
done
