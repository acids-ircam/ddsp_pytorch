# is the same as function_capture.
# deprecate one of the two
#
# binds variables to the function
# by caputring their current value and storing
# them
# let funcA : ()->res
# bind(funcA var1 var2)
# will store var1 and var2 and provide them to the funcA call
function(bind func )
  cmake_parse_arguments("" "" "as" "" ${ARGN})
  if(NOT _as)
    function_new()
    ans(_as)
  endif()

  # if func is not a command import it
  if(NOT COMMAND "${func}")
    function_new()
    ans(original_func)
    function_import("${func}" as ${original_func} REDEFINE)
  else()
    set(original_func "${func}")
  endif()

  set(args ${_UNPARSED_ARGUMENTS})

  set(bound_args)
  foreach(arg ${args})
    set(bound_args "${bound_args}\nset(${arg} \"${${arg}}\")")
  endforeach()

  set(evaluate "function(${_as})
${bound_args}
${original_func}(\${ARGN})    
return_ans()
endfunction()")
  set_ans("")
  eval("${evaluate}")
  return_ref(_as)
endfunction()