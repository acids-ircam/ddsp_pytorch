function(test)


return()
  function(list_iterator_next)


  endfunction()

return()


  map()

    foreach(i RANGE 1 50)
      math(EXPR r "${i} * ${i}")
      kv("${i}" "${r}")
    endforeach()
  end()

  ans(map)
  timer_start(init)
  map_iterator(${map})
  ans(mapit)
  timer_print_elapsed(init)
 
  timer_start(mapit)
  while(true)
    map_iterator_break(mapit)
    message("${mapit.key}: ${mapit.value}")
  endwhile()
  timer_print_elapsed(mapit)

  

return()

  
  function(iterator)
    map_new()
    ans(it)

    set(args ${ARGN})
    

    set(i 0)
    foreach(arg ${args})
      map_set_hidden(${it} ${i} "${arg}")
      math(EXPR i "${i} + 1")
    endforeach()

    set(it ${it};-1;${i};active)
    return_ref(it)
  endfunction()

  macro(iterator_current it_ref)
    list(GET "${it_ref}" 0 iter)
    list(GET "${it_ref}" 1 idx)
    get_property(__ans GLOBAL PROPERTY "${iter}.${idx}")
  endmacro()

  function(iterator_next it_ref)
    list(GET "${it_ref}" 0 iter)
    list(GET "${it_ref}" 1 idx)
    list(GET "${it_ref}" 2 length)

    if("${idx}" STREQUAL "-1")
      set(idx 0)
    else()
      math(EXPR idx "${idx} + 1")
    endif()


    
    list(INSERT "${it_ref}" 1 ${idx}) 
    list(REMOVE_AT "${it_ref}" 2)

    set(end false)
    if(NOT "${idx}" LESS "${length}")
      list(APPEND ${it_ref} end)
      list(REMOVE_ITEM ${it_ref} active)
      set(end true)
    endif()

    set(${it_ref} ${${it_ref}} PARENT_SCOPE)

    return(${end})
  endfunction()


  

  return()

file(GLOB files "C:/Windows/System32/**")
  list(LENGTH files len)

  timer_start(foreac)
  set(res)
  foreach(file ${files})
  #  message("file ${file}")
    string(REGEX MATCH "exe" m "c${file}")
    list(APPEND res ${m})
  endforeach()
  timer_print_elapsed(foreac)

  timer_start(init)
  iterator(${files})
  ans(it)
  timer_print_elapsed(init)
  set(res)
  timer_start(loop)
  while(true)
    iterator_next(it)
    ans(res)
    if(res)
      break()
    endif()
    iterator_current(it)
    ans(current)

    string(REGEX MATCH "exe" m "c${file}")
    list(APPEND res ${m})
  endwhile()
  timer_print_elapsed(loop)
message(len ${len})



  return()


#  timer_start(generate)
  index_range(0 500)
  ans(res)
 # timer_print_elapsed(generate)
 # timer_start(init)
  iterator(${res})
  ans(it)
 # timer_print_elapsed(init)

  #timer_start(asd)
  foreach(i ${res})
    list(GET res "${i}" i2)
    math(EXPR i "${i} + ${i}")
  endforeach()
  #timer_print_elapsed(asd)
  #timer_start(bsd)



  while("${it}" MATCHES "active" )
    iterator_next(it)
    iterator_current(it)
    ans(element)



  endwhile()
  #timer_print_elapsed(bsd)

  return()
  string(ASCII 1 ok)
  

  function(file_stream_open  path)

    return("[eof]")
  endfunction()

  ans(file_stream)



  set(eof "eof")

  while(NOT "${file_stream}" MATCHES "[eof]")
  #  file_stream_read()

  endwhile()

# iterator ref;idx;state



endfunction()