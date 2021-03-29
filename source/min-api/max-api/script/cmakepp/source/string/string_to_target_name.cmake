function(string_to_target_name str)
  string(REGEX REPLACE " \\-\\\\\\/" "_" str "${str}")
  return_ref(str)
endfunction()