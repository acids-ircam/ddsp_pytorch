
# matches the first occurens of regex and returns it
function(regex_search str regex)
  string(REGEX MATCH "${regex}" res "${str}")  
  return_ref(res)
endfunction()