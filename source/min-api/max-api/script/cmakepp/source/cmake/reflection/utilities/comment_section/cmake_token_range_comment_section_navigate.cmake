
## navigates to the specified target sectopn returning its range
## sections are navigated by a simple navigation expression e.g. a.b.c
function(cmake_token_range_comment_section_navigate range path)
  cmake_token_range("${range}")
  ans(range)
  string(REGEX MATCHALL "[^\\.]+" section_identifiers "${path}" )
  foreach(section_identifier ${section_identifiers})
    cmake_token_range_comment_section_find("${range}" "${section_identifier}")
    ans(section)
    if(NOT section)
      error("could not find section '${section_identifier}'")
      return()
    endif() 
    set(range ${section})
  endforeach()

  return_ref(range)
endfunction()