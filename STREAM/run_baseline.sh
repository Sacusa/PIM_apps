declare -a apps_256=("stream_add" "stream_copy" "stream_daxpy" "stream_scale"
    "stream_triad" "bn_fwd" "bn_bwd" "fc")
declare -a apps_1=("kmeans" "histogram")

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <PIM RF size>"
    exit -1
fi

pim_rf_size=$1
outdir=output/pim_rf_size_${pim_rf_size}

source /u/sgupta45/gpgpu-sim_distribution-4.0.1/setup_environment release

mkdir -p ${outdir}

for app in "${apps_256[@]}"; do
    ./main ${app} 1048576 256 > ${outdir}/${app}_256_sm_8 &
done

for app in "${apps_1[@]}"; do
    ./main ${app} 1048576 1 > ${outdir}/${app}_1_sm_8 &
done

./main grim 1048576 32 > ${outdir}/grim_32_sm_8 &

wait
