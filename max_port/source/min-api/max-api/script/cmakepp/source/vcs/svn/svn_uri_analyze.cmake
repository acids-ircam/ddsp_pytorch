  ## svn_uri_analyze(<input:<?uri>> [--revision <rev>] [--branch <branch>] [--tag <tag>])-> 
  ## {
  ##   input: <string>
  ##   uri: <uri string>
  ##   base_uri: <uri string>
  ##   relative_uri: <path>
  ##   ref_type: "branch"|"tag"|"trunk"
  ##   ref: <string>
  ##   revision: <rev>
  ## }
  ##
  ## 
  function(svn_uri_analyze input)
    set(args ${ARGN})

    list_extract_labelled_value(args --revision)
    ans(args_revision)
    list_extract_labelled_value(args --branch)
    ans(args_branch)
    list_extract_labelled_value(args --tag)
    ans(args_tag)

    uri("${input}")
    ans(uri)


    assign(params_revision = uri.params.rev)
    assign(params_branch = uri.params.branch)
    assign(params_tag = uri.params.tag)

    set(trunk_dir trunk)
    set(tags_dir tags)
    set(branches_dir branches)

    uri_format(${uri} --no-query)
    ans(formatted_uri)

    set(uri_revision)
    if("${formatted_uri}" MATCHES "@(([1-9][0-9]*)|HEAD)(\\?|$)")
      set(uri_revision "${CMAKE_MATCH_1}")
      string(REGEX REPLACE "@${uri_revision}" "" formatted_uri "${formatted_uri}")
    endif()

    set(CMAKE_MATCH_3)
    set(uri_ref)
    set(base_uri "${formatted_uri}")
    set(uri_tag)
    set(uri_branch)
    set(uri_rel_path)
    set(uri_ref_type)
    set(ref_type)
    set(ref)
    if("${formatted_uri}" MATCHES "(.*)/(${trunk_dir}|${tags_dir}|${branches_dir})(/|$)")
      set(base_uri "${CMAKE_MATCH_1}")
      set(uri_ref_type "${CMAKE_MATCH_2}")

      set(uri_rel_path "${formatted_uri}")
      string_take(uri_rel_path "${base_uri}/${uri_ref_type}")
      string_take(uri_rel_path "/")

      if(uri_ref_type STREQUAL "${tags_dir}" OR uri_ref_type STREQUAL "${branches_dir}")
        string_take_regex(uri_rel_path "[^/]+")
        ans(uri_ref)
      endif()
      
      if(uri_ref_type STREQUAL "${branches_dir}")
        set(uri_branch ${uri_ref})
      endif()
      if(uri_ref_type STREQUAL "${tags_dir}")
        set(uri_tag "${uri_ref}")
      endif()      

    endif()



    set(revision ${args_revision} ${params_revision} ${uri_revision})
    list_peek_front(revision)
    ans(revision)



    if(uri_ref_type STREQUAL "trunk")
      set(ref_type trunk)
      set(ref trunk)
    endif()

    if(uri_ref_type STREQUAL "branches")
      set(ref_type branch)
      set(ref ${uri_ref})
    endif()

    if(uri_ref_type STREQUAL "tags")
      set(ref_type tag)
      set(ref ${uri_ref})
    endif()

    
    if(args_branch)
      set(ref_type branch)
      set(ref ${args_branch})
    endif()

    if(args_tag)
      set(ref_type tag)
      set(ref ${args_tag})
    endif()

    if("${ref_type}_" STREQUAL "_")
      set(ref_type trunk)
      set(ref)
    endif()


    map_new()
    ans(result)
    map_set(${result} input ${input})
    map_set(${result} uri ${formatted_uri} )
    map_set(${result} base_uri "${base_uri}")
    map_set(${result} relative_uri "${uri_rel_path}")
    map_set(${result} ref_type "${ref_type}")
    map_set(${result} ref "${ref}")
    map_set(${result} revision "${revision}")

    return(${result})
  endfunction()