#declare -a apps=("stream_add" "stream_copy" "stream_daxpy" "stream_scale"
#    "stream_triad")
declare -a apps=("stream_copy" "stream_daxpy" "stream_scale" "stream_triad")

mkdir -p output

for app in "${apps[@]}"; do
    ./main ${app} 67108864 > output/${app}_67108864 &
done

wait
