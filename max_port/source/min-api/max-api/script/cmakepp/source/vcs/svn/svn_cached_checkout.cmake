## svn_cached_checkout()
function(svn_cached_checkout uri)
  set(args ${ARGN})
  path_qualify(target_dir)

  list_extract_flag(args --refresh)
  ans(refresh)
  
  list_extract_flag(args --readonly)
  ans(readonly)


  list_extract_labelled_keyvalue(args --revision)
  ans(revision)
  list_extract_labelled_keyvalue(args --branch)
  ans(branch)
  list_extract_labelled_keyvalue(args --tag)
  ans(tag)

  list_pop_front(args)
  ans(target_dir)
  path_qualify(target_dir)

  
  svn_uri_analyze(${uri} ${revision} ${branch} ${tag})
  ans(svn_uri)

  map_import_properties(${svn_uri} base_uri ref_type ref revision relative_uri)

  if(NOT revision)
    set(revision HEAD)
  endif()


  if("${ref_type}" STREQUAL "branch")
    set(ref_type branches)
  elseif("${ref_type}" STREQUAL "tag")
    set(ref_type tags)
  endif()
  
  cmakepp_config(cache_dir)
  ans(cache_dir)

  string(MD5 cache_key "${base_uri}@${revision}@${ref_type}@${ref}")
  set(cached_path "${cache_dir}/svn_cache/${cache_key}")
  
  if(EXISTS "${cached_path}" AND NOT refresh)
    if(readonly)
      return_ref(cached_path)
    else()
      cp_dir("${cached_path}" "${target_dir}")
      return_ref(target_dir)
    endif()
  endif()

  set(checkout_uri "${base_uri}/${ref_type}/${ref}@${revision}")
  svn_remote_exists("${checkout_uri}")
  ans(remote_exists)
  
  if(NOT remote_exists)
    return()
  endif()


  if(EXISTS "${cached_path}")
    rm("${cached_path}")
  endif()
  mkdir("${cached_path}")


  svn(checkout "${checkout_uri}" "${cached_path}" --non-interactive  --exit-code)
  ans(error)

  if(error)
    rm("${cached_path}")
    return()
  endif()

  if(readonly)
    return_ref(cached_path)
  else()
    cp_dir("${cached_path}" "${target_dir}")
    return_ref(target_dir)
  endif()
endfunction()

