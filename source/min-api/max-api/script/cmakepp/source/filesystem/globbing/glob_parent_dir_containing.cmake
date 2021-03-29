## `(<directory> <glob expression>)-><qualified director>`
## 
## 
## finds the closest parent dir (or dir itself)
## that contains any of the specified glob expressions
## (also see file_glob for syntax)
function(glob_parent_dir_containing )
  glob_up(0 ${ARGN})
  ans(matches)
  list_peek_front(matches)
  ans(first_match)
  if(NOT first_match)
    return()
  endif()

  path_component("${first_match}" PATH)
  ans(first_match)

  return_ref(first_match)
endfunction()