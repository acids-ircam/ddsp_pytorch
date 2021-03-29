# creates a file or updates the file access time
# *by appending an empty string
function(touch path)

  #if("${CMAKE_MAJOR_VERSION}" LESS 3)
    function(touch path)

      path("${path}")
      ans(path)

      set(args ${ARGN})
      list_extract_flag(args --nocreate)
      ans(nocreate)

      if(NOT EXISTS "${path}" AND nocreate)
        return_ref(path)
      elseif(NOT EXISTS "${path}")
        file(WRITE "${path}" "")        
      else()
        file(APPEND "${path}" "")
      endif()


      return_ref(path)

    endfunction()
  touch("${path}")
  return_ans()
endfunction()

