function(test)
  if(NOT WIN32)
    return()
  endif()

  timer_start(t1)
  #reg_query("HKLM/SOFTWARE/Microsoft/Windows/CurrentVersion/Uninstall")
  msi_package_ids()
  ans(res)
  timer_print_elapsed(t1)

  ## at least one package should be installed....
  list(LENGTH res len)
  assert(len)
endfunction()