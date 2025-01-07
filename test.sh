test1(){
log_rootpath=$1
log_prefix="MULTI"
EPOCHS=$2
E=300
for i in $(seq 1 $EPOCHS); do
  seed_num=$((RANDOM))
  for opt in "ADM" "BDA" "BDAG" "RHG" "FHG"; do
    for lr in 0.01 0.05 0.1; do
      echo "$i -> $EPOCHS seed_num:$seed_num opt:$opt lr:$lr"
      python main.py --opt "$opt" -E "$E" --log_prefix "$log_prefix" --log_rootpath "$log_rootpath" --lr "$lr" \
        --seed_num "$seed_num"
    done
  done
done
  #fields = ["opt_method", "hypergrad_method", "rloop0", "time_cost", "last_UL_dval", "last_LL_dval", "orth_x", "orth_y"]
python main_auxx.py "$log_rootpath" "$log_prefix" opt rloop0 time_cost last_UL_dval last_LL_dval lr E
}

test(){
log_rootpath=$1
log_prefix="BDA"
EPOCHS=$2
E=300
for i in $(seq 1 $EPOCHS); do
  seed_num=$((RANDOM))
  for mu in $(seq 0 0.2 1); do
    for lr in 0.01 0.1; do
        for orth_y in "True" "False"; do
          for loop1 in $(seq 1 3 20); do
            for hyme in "backward" "forward"; do
              echo "$i -> $EPOCHS seed_num:$seed_num mu:$mu lr:$lr ox:True oy:$orth_y l1:$loop1 hyme:$hyme"
              python main.py --opt_method "BDA" --hypergrad_method "$hyme" -E "$E" --log_prefix "$log_prefix" \
                --mu "$mu" --orth_x True --orth_y "$orth_y" --seed_num "$seed_num" --loop1 "$loop1" \
                --log_rootpath "$log_rootpath" --lr "$lr"
            done
          done
      done
    done
  done
done
python main_auxx.py "$log_rootpath" "$log_prefix" opt_method hypergrad_method rloop0 time_cost last_UL_dval last_LL_dval orth_x orth_y mu loop1 lr
}