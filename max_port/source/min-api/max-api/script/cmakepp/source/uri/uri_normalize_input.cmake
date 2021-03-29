
## normalizes the input for the uri
## expects <uri> to have a property called input
## ensures a property called uri is added to <uri> which contains a valid uri string 
function(uri_normalize_input input_uri)
  set(flags ${ARGN})


  # options  
  set(handle_windows_paths true)
  set(default_file_scheme true)
  set(driveletter_separator :)
  set(delimiters "''" "\"\"" "<>")
  set(encode_input 32) # character codes to encode in delimited input
  set(ignore_leading_whitespace true)
  map_get("${input_uri}" input)
  ans(input)

  if(ignore_leading_whitespace)
    string_take_whitespace(input)
  endif()

  set(delimited)
  foreach(delimiter ${delimiters})
    string_take_delimited(input "${delimiter}")
    ans(delimited)
    if(NOT "${delimited}_" STREQUAL "_")
      break()
    endif()
  endforeach()

  set(delimiters "${delimiter}")

    # if string is delimited encode whitespace 
    if(NOT "${delimited}_" STREQUAL "_")
      set(rest "${input}")
      set(input "${delimited}")
      
      if(ignore_leading_whitespace)
        string_take_whitespace(input)
      endif()

      if(encode_input)
        uri_encode("${input}" 32)
        ans(input)
      endif()
    endif()

    

    # the whole uri is delimited by a space or end of string
    set(CMAKE_MATCH_1)
    set(CMAKE_MATCH_2)
    set(uri)
    if("_${input}" MATCHES "^_(${uric}+)(.*)")
      set(uri "${CMAKE_MATCH_1}")
      set(input "${CMAKE_MATCH_2}")
    endif()
    #string_take_regex(input "${uric}+")
    #ans(uri)

    if("${rest}_" STREQUAL "_")
      set(rest "${input}")
    endif()


    set(windows_absolute_path false)
    if(default_file_scheme)
      if(handle_windows_paths)
        # replace backward slash with forward slash
        # for windows paths - non standard behaviour
        string(REPLACE \\ /  uri "${uri}")
      endif()  


      if("_${uri}" MATCHES "^_/" AND NOT "_${uri}" MATCHES "^_//")
        set(uri "file://${uri}")
      endif()

      if("_${uri}" MATCHES "^_[a-zA-Z]:")
        #local windows path no scheme -> scheme is file://
        # <drive letter>: is replaced by /<drive letter>|/
        # also colon after drive letter is normalized to  ${driveletter_separator}
        string(REGEX REPLACE "^_([a-zA-Z]):(.+)" "\\1${driveletter_separator}\\2" uri "_${uri}")
        set(uri "file:///${uri}")
        set(windows_absolute_path true)
      endif()

    endif()
    
    # the rest is not part of input_uri
    map_capture(${input_uri} uri rest delimited_rest delimiters windows_absolute_path)
    return_ref(input_uri)

endfunction()
