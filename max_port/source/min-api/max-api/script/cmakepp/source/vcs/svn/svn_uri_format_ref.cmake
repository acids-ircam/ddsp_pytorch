
  function(svn_uri_format_ref svn_uri)
    map_import_properties(${svn_uri} base_uri revision ref ref_type)

    string(REGEX REPLACE "^svnscm\\+" "" base_uri "${base_uri}")
    if(NOT revision)
      set(revision HEAD)
    endif()

    if("${ref_type}" STREQUAL "branch")
      set(ref_type branches)
    elseif("${ref_type}" STREQUAL "tag")
      set(ref_type tags)
    endif()
    
    set(checkout_uri "${base_uri}/${ref_type}/${ref}@${revision}")
    return_ref(checkout_uri)

  endfunction()
