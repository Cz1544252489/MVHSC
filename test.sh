test1(){
log_rootpath="test1"
log_prefix="ADM2_orth"
EPOCHS=20
E=200
for i in $(seq 1 $EPOCHS); do
  seed_num=$((RANDOM))
  echo "$i -> $EPOCHS $seed_num"
  python main.py --opt_method "ADM" --log_prefix "${log_prefix}" \
   --seed_num "$seed_num" -E "$E" --log_rootpath "$log_rootpath"
  python main.py --opt_method "ADM" --orth_x False --log_prefix "${log_prefix}" \
   --seed_num "$seed_num" -E "$E" --log_rootpath "$log_rootpath"
  python main.py --opt_method "ADM" --orth_y False --log_prefix "${log_prefix}" \
   --seed_num "$seed_num" -E "$E" --log_rootpath "$log_rootpath"
  python main.py --opt_method "ADM" --orth_x False --orth_y False --log_prefix "${log_prefix}" \
   --seed_num "$seed_num" -E "$E" --log_rootpath "$log_rootpath"
done
  #fields = ["opt_method", "hypergrad_method", "rloop0", "time_cost", "last_UL_dval", "last_LL_dval", "orth_x", "orth_y"]
python main_auxx.py "$log_prefix" opt_method rloop0 time_cost last_UL_dval last_LL_dval orth_x orth_y
}

test2(){
log_rootpath="test2"
log_prefix="BDA_F_orth"
EPOCHS=20
E=100
for i in $(seq 1 $EPOCHS); do
  seed_num=$((RANDOM))
  echo "$i -> $EPOCHS $seed_num"
  python main.py --opt_method "BDA" --hypergrad_method "forward" --log_prefix "${log_prefix}" \
  --seed_num "$seed_num" -E "$E" --log_rootpath "$log_rootpath"
  python main.py --opt_method "BDA" --hypergrad_method "forward" --log_prefix "${log_prefix}" \
   --orth_y False --seed_num "$seed_num" -E "$E" --log_rootpath "$log_rootpath"
  python main.py --opt_method "BDA" --hypergrad_method "forward" --log_prefix "${log_prefix}" \
   --orth_x False --seed_num "$seed_num" -E "$E" --log_rootpath "$log_rootpath"
  python main.py --opt_method "BDA" --hypergrad_method "forward" --log_prefix "${log_prefix}" \
   --orth_x False --orth_y False --seed_num "$seed_num" -E "$E" --log_rootpath "$log_rootpath"
done
python main_auxx.py "$log_prefix" opt_method hypergrad_method rloop0 time_cost last_UL_dval last_LL_dval orth_x orth_y
}

test3(){
log_rootpath="test3"
log_prefix="BDA_B_orth2"
EPOCHS=20
E=300
for i in $(seq 1 $EPOCHS); do
  seed_num=$((RANDOM))
  echo "$i -> $EPOCHS $seed_num"
  python main.py --opt_method "BDA" --hypergrad_method "backward" --log_prefix "${log_prefix}" \
   --seed_num "$seed_num" -E "$E" --log_rootpath "$log_rootpath"
  python main.py --opt_method "BDA" --hypergrad_method "backward" --log_prefix "${log_prefix}" \
   --orth_y False --seed_num "$seed_num" -E "$E" --log_rootpath "$log_rootpath"
  python main.py --opt_method "BDA" --hypergrad_method "backward" --log_prefix "${log_prefix}" \
   --orth_x False --seed_num "$seed_num" -E "$E" --log_rootpath "$log_rootpath"
  python main.py --opt_method "BDA" --hypergrad_method "backward" --log_prefix "${log_prefix}" \
   --orth_x False --orth_y False --seed_num "$seed_num" -E "$E" --log_rootpath "$log_rootpath"
done
python main_auxx.py "$log_prefix" opt_method hypergrad_method rloop0 time_cost last_UL_dval last_LL_dval orth_x orth_y
}
