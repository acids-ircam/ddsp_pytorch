  function(graphsearch)
    cmake_parse_arguments("" "" "EXPAND;PUSH;POP" "" ${ARGN})

    if(NOT _EXPAND)
      message(FATAL_ERROR "graphsearch: no expand function set")
    endif()

    function_import("${_EXPAND}" as gs_expand REDEFINE)
    function_import("${_PUSH}" as gs_push REDEFINE)
    function_import("${_POP}" as gs_pop REDEFINE)

    # add all arguments to stack
    foreach(node ${_UNPARSED_ARGUMENTS})

      gs_push(${node})
    endforeach()

    # iterate
    while(true)
      gs_pop()
      ans(current)
      #message("current ${current}")
      # recursion anchor - no more node
      if(NOT current)
        break()
      endif()
      gs_expand(${current})
      ans(successors)
      foreach(successor ${successors})
        gs_push(${successor})
      endforeach()
      
    endwhile()
  endfunction()
