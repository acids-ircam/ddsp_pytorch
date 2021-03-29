
function(alias_remove name)
  path("${CMAKE_CURRENT_LIST_DIR}/../bin")
  ans(path)
  if(WIN32)
    file(REMOVE "${path}/${name}.bat")
  else()
    message(FATAL_ERROR "only implemnted for windows")
  endif()

endfunction()
