
function(linux_ps_info pid key)
  linux_ps_lean(-p "${pid}" -o "${key}=")
  ans_extract(error)
  ans(stdout)
  #print_vars(error stdout)

  if(error)
    return()
  endif()
  string(STRIP "${stdout}" val)
  return_ref(val)
endfunction()