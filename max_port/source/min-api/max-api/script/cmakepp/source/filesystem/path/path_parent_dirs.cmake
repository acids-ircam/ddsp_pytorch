# returns the specified max n (all if n = 0)
# parent directories of path
function(path_parent_dirs path)
  set(continue 99999)
  if(ARGN )
    set(continue "${ARGN}")

    if("${continue}" EQUAL 0)
      set(continue 99999)
    endif()
  endif()

  path("${path}")
  ans(path)

  set(isrooted false)
  if("_${path}" MATCHES "^_[/]")
    set(isrooted true)
  endif()

  path_split("${path}")
  ans(parts)


  set(parent_dirs)
  while(true)
    if(NOT parts OR ${continue} LESS 1)
      break()
    endif()
    list_pop_back(parts)
    path_combine(${parts})
    ans(current)      

    if(isrooted)
      set(current "/${current}")
    endif()
    
    if("_${current}" STREQUAL "_")
      break()
    endif()
    list(APPEND parent_dirs "${current}")
    math(EXPR continue "${continue} - 1")

  endwhile()
  return_ref(parent_dirs)
endfunction()

