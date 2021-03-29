# reads the file specified and returns its content
function(fread_lines path)
  path_qualify(path)
  set(args ${ARGN})

  list_extract_labelled_keyvalue(args --regex REGEX)
  ans(regex)
  list_extract_labelled_keyvalue(args --limit-count LIMIT_COUNT)
  ans(limit_count)
  list_extract_labelled_keyvalue(args --limit-input LIMIT_INPUT)
  ans(limit_input)
  list_extract_labelled_keyvalue(args --limit-output LIMIT_OUTPUT)
  ans(limit_output)
  list_extract_labelled_keyvalue(args --length-minimum LENGTH_MINIMUM)
  ans(length_minimum)
  list_extract_labelled_keyvalue(args --length-maximum LENGTH_MAXIMUM)
  ans(length_maximum)
  list_extract_flag_name(args --newline-consume NEWLINE_CONSUME)
  ans(newline_cosume)
  list_extract_flag_name(args --no-hex-conversion NO_HEX_CONVERSION)
  ans(no_hex_conversion)


  file(STRINGS "${path}" res 
    ${limit_count} 
    ${limit_input} 
    ${limit_output} 
    ${length_minimum} 
    ${length_maximum}
    ${newline_cosume}
    ${regex}
    ${no_hex_conversion}
  )

  return_ref(res)
endfunction()