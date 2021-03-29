# returns an object from string, or reference
# ie obj("{id:1, test:'asd'}") will return an object
  function(obj object_ish)
    is_map("${object_ish}")
    ans(isobj)
    if(isobj)
      return("${object_ish}")
    endif()
    if("${object_ish}" MATCHES "^{.*}$")
     script("${object_ish}")
     return_ans()
    endif()
    return()
  endfunction()