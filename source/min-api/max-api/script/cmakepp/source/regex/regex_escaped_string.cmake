

function(regex_escaped_string delimiter_begin delimiter_end)

  set(regex "${delimiter_begin}(([^${delimiter_end}\\]|([\\][${delimiter_end}])|([\\][\\])|([\\]))*)${delimiter_end}")
  return_ref(regex)
endfunction()