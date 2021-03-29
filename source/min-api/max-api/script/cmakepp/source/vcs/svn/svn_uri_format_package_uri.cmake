
  function(svn_uri_format_package_uri svn_uri)
    map_import_properties(${svn_uri} base_uri revision ref ref_type)

    string(REGEX REPLACE "^svnscm\\+" "" base_uri "${base_uri}")

    if("${ref_type}" STREQUAL "branch")
      set(ref_type branches)
    elseif("${ref_type}" STREQUAL "tag")
      set(ref_type tags)
    endif()

    if(revision STREQUAL "HEAD")
      set(revision)
    endif() 


    set(params)
    if(NOT ref_type STREQUAL "trunk" OR revision)
      map_new()
      ans(params)
      if(NOT revision STREQUAL "")
        map_set(${params} rev "${revision}")
      endif()
      if(ref_type STREQUAL trunk)
      elseif("${ref_type}" STREQUAL "branch")
        map_set(${params} branch "${ref}")
      elseif("${ref_type}" STREQUAL "tag")
        map_set(${params} branch "${ref}")
      endif()
      uri_params_serialize(${params})
      ans(query)
      set(query "?${query}")
    endif()

    set(result "${base_uri}${query}")


    return_ref(result)


  endfunction()