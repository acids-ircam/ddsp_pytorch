
  function(string_indent str indentation)
    string_codes()

    set(maxWidth ${ARGN})
    if("${maxWidth}_" STREQUAL "_")
      set(maxWidth 0)
    endif()
    
    # normalize line endings
    string(REPLACE "\r\n" "\n"  str "${str}")

    if("${maxWidth}" LESS 1)
      string(REPLACE "\n" "\n${indentation}" str "${str}")
      set(str "${indentation}${str}")
      return_ref(str)      
    endif()

    # desemicolonize
    string(REPLACE ";" "${semicolon_code}" str "${str}")

    #string(REPLACE "\n" ";" str "${str}")

    string(REPLACE " " ";" str "${str}")
    string(REPLACE "\n" ";${free_token1}" str "${str}")
    set(result)

    set(currentLine)
    set(currentLength 0)
    ## 
    while(true)
     list(LENGTH str size)
      if(NOT size)
        if(NOT "${currentLine}_" STREQUAL "_")
          list(APPEND result "${currentLine}")
        endif()
        break()
      endif()

      list_pop_front(str)
      ans(word)

      if("${word}" STREQUAL "${free_token1}")
        list(APPEND result "${currentLine}")
        set(currentLine)
        continue()
      endif()

     
      set(currentLine "${currentLine} ${word}")
      string(LENGTH "${currentLine}" len)
      if(NOT "${len}" LESS "${maxWidth}")
        list(APPEND result "${currentLine}")
        set(currentLine)
      endif()
    endwhile()

    string(REPLACE "${free_token1}" "\n" result "${result}")

    string(REPLACE ";" "\n" result "${result}")
    string(REPLACE "\n" "\n${indentation}" result "${result}")
    string(REPLACE "${semicolon_code}" ";" result "${result}")
    set(result "${indentation}${result}")

    return_ref(result)
  endfunction()