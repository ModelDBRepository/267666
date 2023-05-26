run() {
    ii=$1

        python runMultiProcs.py $ii;
}

num_processes=10

for ii in `seq 0 200`;
do
    ((i=i%num_processes)); ((i++==0)) && wait
    run $ii &
done
