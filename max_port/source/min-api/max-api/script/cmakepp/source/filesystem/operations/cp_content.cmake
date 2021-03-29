  ## cp_content(<source dir> <target dir> <glob ignore expression...>) -> <path...> 
  ## 
  ## copies the content of source dir to target_dir respecting 
  ## the globging expressions if none are given
  ## returns the copied paths if globbing expressiosnw were used
  ## else returns the qualified target_dir
  function(cp_content source_dir target_dir)

    path_qualify(target_dir)
    path_qualify(source_dir)
    set(content_globbing_expression ${ARGN})
    if(NOT content_globbing_expression)
      cp_dir("${source_dir}" "${target_dir}")
      ans(res)
    else()
        pushd("${source_dir}")
            cp_glob("${target_dir}" ${content_globbing_expression})
            ans(res)
        popd()
    endif()
    return_ref(res)
  endfunction()