##
## compiles a lambda expression to valid cmake source and returns it
## {{a}} -> ${a}
## ["["<capture>"]"]["("<arg defs>")"] [(<expression>";")*]
## 
## 
function(lambda2_compile source)
  string_encode_list("${source}")
  ans(source)
  string_codes()
  regex_cmake()

  string_take_whitespace(source)

  set(capture_group_regex "${bracket_open_code}([^${bracket_close_code}]*)${bracket_close_code}")
  if("${source}" MATCHES "^(${capture_group_regex})(.*)")
    set(capture "${CMAKE_MATCH_2}")
    set(source "${CMAKE_MATCH_3}")
    string(REPLACE " " ";" capture "${capture}")
  else() 
    set(capture)
  endif()

  string_take_whitespace(source)
  if("${source}" MATCHES "^\\(([^\\)]*)\\)(.*)")
    set(signature "${CMAKE_MATCH_1}")
    set(source "${CMAKE_MATCH_2}")
  else()

  endif()



  string_take_whitespace(source)

  lambda2_compile_source("${source}")
  ans(cmake_source)
    




  map_capture_new(signature capture source cmake_source)

  return_ans()

endfunction()
