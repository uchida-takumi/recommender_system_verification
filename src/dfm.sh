for k_hold in {0..59}
do
  echo "python3 src/dfm_data_set.py --$k_hold"
  #python3 src/dfm_data_set.py --$i $n_hold
  #python3 src/DeepFM_A01_get_pickle_for_validation.py
done
