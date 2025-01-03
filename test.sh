test1(){
log_prefix="ADM_orth"
for i in $(seq 1 100); do
  echo "$i"
  python main.py --opt_method "ADM" --log_prefix "${log_prefix}"
  python main.py --opt_method "ADM" --orth_x False --log_prefix "${log_prefix}"
  python main.py --opt_method "ADM" --orth_y False --log_prefix "${log_prefix}"
  python main.py --opt_method "ADM" --orth_x False --orth_y False --log_prefix "${log_prefix}"
done
  #fields = ["opt_method", "hypergrad_method", "rloop0", "time_cost", "last_UL_dval", "last_LL_dval", "orth_x", "orth_y"]
python main_auxx.py "$log_prefix" opt_method rloop0 time_cost last_UL_dval last_LL_dval orth_x orth_y
}