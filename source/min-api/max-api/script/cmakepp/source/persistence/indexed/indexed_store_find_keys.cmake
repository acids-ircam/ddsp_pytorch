
  function(indexed_store_find_keys)
    set(keys)
    this_get(store_dir)

    set(globs)
    foreach(query ${ARGN})
      checksum_string("${query}")
      ans(hash)
      list(APPEND globs "${store_dir}/${hash}*")
    endforeach()
    if(NOT globs)
      return()
    endif()

    file(GLOB store_keys RELATIVE "${store_dir}" ${globs})
    string(REGEX REPLACE "([a-fA-F0-9]+)-([a-fA-F0-9]+)-([a-fA-F0-9]+)" "\\2" keys "${store_keys}")
  
    list_remove_duplicates(keys)
    return_ref(keys)
  endfunction()
  