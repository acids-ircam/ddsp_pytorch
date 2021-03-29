## `(<command invocation token> <values: <any...>>)-><void>`
##
## replaces the arguments for the specified invocation by the
## specified values. The values are quoted if necessary
function(cmake_invocation_token_set_arguments invocation_token)
  cmake_invocation_get_arguments_range("${invocation_token}")
  ans_extract(begin end)

  cmake_arguments_quote_if_necessary(${ARGN})
  ans(arguments)

  list(LENGTH ARGN count)
  string(LENGTH "${ARGN}" len)
  

  if(${len} LESS 70 OR ${count} LESS 2)
    string_combine(" " ${arguments})
    ans(argument_string)
  else()
    map_tryget(${invocation_token} column)
    ans(column)
    string_repeat(" " "${column}")
    ans(last_indentation)
    math(EXPR column "${column} + 2")
    string_repeat(" " "${column}")
    ans(indentation)
    string_combine("\n${indentation}" ${arguments})
    ans(argument_string)
    set(argument_string "${argument_string}\n${last_indentation}")
  endif()

  cmake_token_range("${argument_string}")
  ans(argument_token_range)


  cmake_token_range_replace("${begin};${end}" "${argument_token_range}")
  return()
endfunction()