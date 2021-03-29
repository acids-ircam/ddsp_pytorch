## `(<line number>)-><cmake code>`
##
## compiles all expressions in the current scope (file or function()...endfunction())
function(cmakepp_compile_scope_expressions line)
  ## read the current list file (calling file)
  fread("${CMAKE_CURRENT_LIST_FILE}")
  ans(cmake_code)

  ## convert code to tokens
  cmake_token_range("${cmake_code}")
  rethrow()
  ans_extract(begin end)

  ## get the invocation from which called this function
  cmake_invocation_filter_token_range("${begin};${end}" \${line} EQUAL ${line})
  ans(invocation)

  if(NOT invocation)
    throw("could not find invocation at ${CMAKE_CURRENT_LIST_FILE}:${line}")
  endif()

  ## get the first enabled token (after the invocation)
  map_tryget("${invocation}" arguments_end_token)
  ans(first_enabled_token)
  map_tryget("${first_enabled_token}" next)
  ans(first_enabled_token)



  set(current_begin ${first_enabled_token})  
  set(scope_depth 0)
  ## loop through all invocations. and count function/endfunctions
  ## break when end of scope found 
  while(true)
    cmake_invocation_filter_token_range("${current_begin};" \${invocation_identifier} MATCHES "^(endfunction)|(function)$" --take 1)
    ans(invocation)
    if(NOT invocation)
      break()
    endif()
    map_tryget("${invocation}" invocation_identifier)
    ans(invocation_identifier)
    ## extendable:  macro/endmacro while/endwhile (begin/end)
    if("${invocation_identifier}" STREQUAL "function" )
      math(EXPR scope_depth "${scope_depth} + 1")
    elseif("${invocation_identifier}" STREQUAL "endfunction" )
      math(EXPR scope_depth "${scope_depth} - 1")
      if(${scope_depth} LESS 0)
        map_tryget("${invocation}" invocation_token)
        ans(last_enabled_token)
        break()
      endif()
    endif()
    map_tryget("${invocation}" arguments_end_token)
    ans(current_begin)
  endwhile()

  ## get the code in the expression enabled range and compile it
  cmake_token_range_serialize("${first_enabled_token};${last_enabled_token}")
  ans(enabled_code)
  cmakepp_expr_compile("${enabled_code}")
  ans(result)

  ## return the compiled code
  return_ref(result)
endfunction()