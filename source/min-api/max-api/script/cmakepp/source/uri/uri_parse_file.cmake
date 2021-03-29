## expects last_segment property to exist
## ensures file_name, file, extension exists
function(uri_parse_file uri)
  map_get("${uri}" last_segment)
  ans(file)

  if ("_${file}" MATCHES "\\.") # file contains an extension
      string(REGEX MATCH "[^\\.]+$" extension "${file}")
      string(LENGTH "${extension}" extension_length)

      if (extension_length)
          math(EXPR extension_length "0 - ${extension_length}  - 2")
          string_slice("${file}" 0 ${extension_length})
          ans(file_name)
      endif ()
  else ()
      set(file_name "${file}")
      set(extension "")
  endif ()
  map_capture(${uri} file extension file_name)
endfunction()