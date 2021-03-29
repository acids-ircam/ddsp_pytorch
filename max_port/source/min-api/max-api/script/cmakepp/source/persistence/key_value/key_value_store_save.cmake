
  function(key_value_store_save)
    this_get(store_dir)
    assign(key = this.key(${ARGN}))
    qm_write("${store_dir}/${key}" ${ARGN})    
    return_ref(key)
  endfunction()