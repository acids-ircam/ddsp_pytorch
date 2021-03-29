

  function(lambda2_instanciate source)

    lambda2_compile("${source}")
    ans(lambda)

    map_tryget(${lambda} capture)
    ans(captures)
    set(capture_code)    
    foreach(capture ${captures})
      set(capture_code "${capture_code}\n  set(${capture} \"${${capture}}\")")
    endforeach()


    set(function_name ${ARGN})
    if(NOT function_name)
      function_new()
      ans(function_name)
    endif()
    map_set(${lambda} function_name ${function_name})

    map_tryget(${lambda} cmake_source)
    ans(cmake_source)
    map_tryget(${lambda} signature)
    ans(signature) 
    set(source "function(${function_name} ${signature})${capture_code}\n${cmake_source}\nendfunction()")
    eval("${source}")
    map_set(${lambda} cmake_function "${source}")
    return_ref(lambda)
  endfunction()
