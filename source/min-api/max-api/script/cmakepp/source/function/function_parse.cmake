
function(function_parse function_ish)
  is_function(function_type "${function_ish}")
  if(NOT function_type)
    return()
  endif()
  function_string_get( "${function_ish}")
  ans(function_string)
  
  if(NOT function_string)
    return()
  endif()

  function_signature_regex(regex)
  function_signature_get( "${function_string}")
  ans(signature)

  string(REGEX REPLACE ${regex} "\\1" func_type "${signature}" )
  string(REGEX REPLACE ${regex} "\\2" func_name "${signature}" )
  string(REGEX REPLACE ${regex} "\\3" func_args "${signature}" )

  string(STRIP "${func_name}" func_name)

  # get args
  string(FIND "${func_args}" ")" endOfArgsIndex)
  string(SUBSTRING "${func_args}" "0" "${endOfArgsIndex}" func_args)

  if(func_args)
    string(REGEX MATCHALL "[A-Za-z0-9_\\\\-]+" all_args ${func_args})
  endif()

  string(SUBSTRING "${func_args}" 0 ${endOfArgsIndex} func_args)
  string(TOLOWER "${func_type}" func_type)


  map_new()
  ans(res)
  map_set(${res} type "${func_type}")
  map_set(${res} name "${func_name}")
  map_set(${res} args "${all_args}")
  map_set(${res} code "${function_string}")

  return(${res})
endfunction()

