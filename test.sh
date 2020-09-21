#!/bin/sh

n_atom=5
for i in $(seq 1 $n_atom); do
    scaling_opt=${scaling_opt}"--scaling "$i" "$((i+1))" "
done

echo $scaling_opt