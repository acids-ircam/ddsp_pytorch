
  # splits the speicifed path into its directories and files
  # e.g. c:/dir1/dir2/file.ext -> ['c:','dir1','dir2','file.ext'] 
  function(path_split path)
    if("_${path}" MATCHES "^_[\\/]")
      string_substring("${path}" 1)
      ans(path)
    endif()
    string_split("${path}" "[/]")
    ans(parts)

    return_ref(parts)
  endfunction()