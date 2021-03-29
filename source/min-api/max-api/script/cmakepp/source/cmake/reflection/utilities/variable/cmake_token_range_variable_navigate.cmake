## navigates to and tries to change variable
function(cmake_token_range_variable_navigate range variable_path)
  cmake_token_range("${range}")
  ans(range)
  set(args ${ARGN}) 

  string(REGEX MATCH "[^\\.]+$" variable_name "${variable_path}")
  string(REGEX REPLACE "\\.?[^\\.]+$" "" section_path "${variable_path}" )
  cmake_token_range_comment_section_navigate("${range}" "${section_path}")
  ans(section)

  if(NOT section)
    return()  
  endif()
  cmake_token_range_variable("${section}" "${variable_name}" ${args})
  return_ans()  
endfunction()