declare -a apps=("stream_add" "stream_copy" "stream_daxpy" "stream_scale"
    "stream_triad" "bn_fwd" "bn_bwd" "fc" "kmeans" "grim")

mkdir -p output

for app in "${apps[@]}"; do
    ./main ${app} 67108864 > output/${app}_67108864 &
done

wait
