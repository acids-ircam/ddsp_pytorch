
  ## tries to interpret the uri as a local path and replaces it 
  ## with a normalized local path (ie file:// ...)
  ## returns a new uri
  function(uri_qualify_local_path uri)
    uri("${uri}")
    ans(uri)

    map_tryget(${uri} input)
    ans(uri_string)

    map_tryget(${uri} normalized_host)
    ans(normalized_host)

    map_tryget("${uri}" scheme)
    ans(scheme)


    ## check if path path is going to be local
    eval_truth(
       "${scheme}_" MATCHES "(^_$)|(^file_$)" # scheme is file
       AND normalized_host STREQUAL "localhost" # and host is localhost 
       AND NOT "${uri_string}" MATCHES "^[^/]+:" # and input uri is not scp like ssh syntax
     ) 
    ans(is_local)

    ## special handling of local path
    if(is_local)
      ## use the locally qualfied full path
      map_get("${uri}" path)
      ans(local_path)
      path_qualify(local_path)
      map_tryget(${uri} params)
      ans(params)
      uri("${local_path}")
      ans(uri)
      map_set("${uri}" params "${params}")
    endif()
    return_ref(uri)
  endfunction()