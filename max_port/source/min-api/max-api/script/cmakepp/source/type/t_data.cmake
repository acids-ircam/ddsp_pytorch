function(t_data)
  
  fopen_data("${ARGN}")
  ans(data)

  if(data)
    set(data true ${data})
    return_ref(data)
  endif()

  data(${ARGN})
  ans(data)

  if(data)
    set(data true ${data})
    return_ref(data)
  endif()

  return(false)
endfunction()