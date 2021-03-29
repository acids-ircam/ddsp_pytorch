
function(markdown_see_function function_sig)
  if("${function_sig}" MATCHES "^([a-zA-Z0-9_]+)[ \\t]*\\(")
    set(function_name "${CMAKE_MATCH_1}")

    return("[`${function_sig}`](#${function_name})")
  endif()
  return()
endfunction()