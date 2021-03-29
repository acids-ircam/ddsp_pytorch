function(test)




  macro(goto label)
    _message("going to ${label}")
    if(NOT __goingto)
      message("first goto")
      set(file "${CMAKE_CURRENT_LIST_FILE}")
      set(__goingto "${file}")
    else()
      message("nested goto")
      set(file "${__goingto}")
    endif()
    is_temp_file("${file}")
    ans(is_temp_file)
    if(is_temp_file)
      message(FATAL_ERROR "goto not possible in temp_file")
    endif()
    fread("${file}")
    ans(content)
    cmake_token_range("${content}")
    ans_extract(begin end)


    cmake_invocation_filter_token_range("${begin};${end}"  --take 1
      \${invocation_identifier} STREQUAL label
      )
    ans(label_token)
    if(NOT label_token)
      message(FATAL_ERROR "label not found in current scope")
    endif()

    map_tryget("${label_token}" arguments_end_token)
    ans(arg_end)
    map_tryget("${arg_end}" next)
    ans(arg_end)
    # cmake_invocation_filter_token_range("${label}${end}" --take 1
    #   \${invocation_identifier} MATCHES "^(endfunction)|(endmacro)|(function)|(endfunction)$"
    #   )
set(parent_return_token ${return_token})
  address_new()
  ans(return_token)
    cmake_token_range_serialize("${arg_end};${end}")
    ans(rest)
    set(rest "${rest}\n address_set(${return_token} true)\n")
   # _message("${rest}")

    ## get the the rest invocations of current scope (file, function or macro)
    ## add a mechanism which returns to the original position af
    ## put into a file and eval


    eval_ref(rest)

    address_get("${return_token}")
    ans(has_returned)

    if(has_returned)
      _return()
    endif()
  endmacro()

  function(label)
    
  endfunction()


fwrite("test.cmake" "

set(i 1)
label(loop)
_message(\"iteration: \${i}\")
if(\${i} LESS 5)
  math(EXPR i \"\${i} + 1\")
  goto(loop)
endif()


message(ohoh)




")
include("${test_dir}/test.cmake")



return()

endfunction()