##
##
##
function(eval_predicate)
  arguments_encoded_list(0 ${ARGC})
  ans(__eval_predicate_arguments)

  ## slower version that works
    encoded_list_to_cmake_string(${__eval_predicate_arguments})
    ans(__eval_predicate_predicate)

    set(__eval_predicate_code "
      if(${__eval_predicate_predicate})
        set(__ans true)
      else()
        set(__ans false)
      endif()
    ")
    eval("${__eval_predicate_code}")
    

    set(__ans "${__ans}" PARENT_SCOPE)
_return() 


  ##womething is wrong here
  is_encoded_list(${__eval_predicate_arguments})
  ans(__eval_predicate_is_encoded_list)
  if(NOT __eval_predicate_is_encoded_list)
    __eval_predicate_helper("${__eval_predicate_arguments}")
    ans(__eval_predicate_arguments)

    if(${__eval_predicate_arguments})
      set(__ans true PARENT_SCOPE)
    else()
      set(__ans false PARENT_SCOPE)
    endif()
  else()
    encoded_list_to_cmake_string(${__eval_predicate_arguments})
    ans(__eval_predicate_predicate)

    set(__eval_predicate_code "
      if(${__eval_predicate_predicate})
        set(__ans true)
      else()
        set(__ans false)
      endif()
    ")
    eval("${__eval_predicate_code}")
    set(__ans ${__ans} PARENT_SCOPE)
  endif()
  _message("${__eval_predicate_arguments} (${__eval_predicate_is_encoded_list}): ${__ans}")
endfunction()



macro(__eval_predicate_helper)
  set(__ans "${ARGN}")
endmacro()