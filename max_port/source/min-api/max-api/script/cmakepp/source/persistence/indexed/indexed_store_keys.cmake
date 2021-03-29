
function(indexed_store_keys)
  this_get(store_dir)
  file(GLOB keys RELATIVE "${store_dir}" "${store_dir}/*" )
  string(REGEX REPLACE "[a-fA-F0-9]+-([a-fA-F0-9]+)-[a-fA-F0-9]+" "\\1" keys "${keys}")
  
  list_remove_duplicates(keys)
  return_ref(keys)
endfunction()