
  function(define_http_resource function uri_string)
    uri("${uri_string}")
    ans(uri)

    map_tryget("${uri}" scheme_specific_part)
    ans(scheme_specific_part)
    map_tryget("${uri}" scheme)
    ans(scheme)

    string(REGEX MATCHALL ":([a-zA-Z][a-zA-Z0-9_]*)" replaces "${scheme_specific_part}")

    list_remove_duplicates(replaces)

    set(function_args "")

    foreach(replace ${replaces})
      string(REGEX REPLACE ":([a-zA-Z][a-zA-Z0-9_]*)" "\\1" name "${replace}")
      string(REPLACE "${replace}" "\${${name}}" uri_string "${uri_string}")
      set(function_args "${function_args} ${name}")
    endforeach()    

    set(code "
      function(${function}${function_args})
        set(args \${ARGN})
        list_extract_flag(args --put)
        ans(put)
        set(resource_uri \"${uri_string}\")

        if(put)
          http_put(\"\${resource_uri}\" \${args})
        else()
          http_get(\"\${resource_uri}\" \${args})
        endif()
        return_ans()
      endfunction()
    ")
    eval("${code}")
    return()

  endfunction()