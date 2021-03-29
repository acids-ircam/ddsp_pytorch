function(test)
  mkdir("${test_dir}")
  cd("${test_dir}")
  function(get_property_list)
    cmake(--help-property-list --process-handle)
    ans(res)
    map_tryget(${res} stdout)
    ans(stdout)
    string(REPLACE "\n" ";" stdout "${stdout}")
    list_pop_front(stdout)

    
    return(${stdout})
  endfunction()

  get_property_list()
  ans(res)


endfunction()