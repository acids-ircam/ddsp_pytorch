
  ## cp_glob(<target dir> <glob. ..> )-> <path...>
  ##
  ## 
  function(cp_glob target_dir)
    set(args ${ARGN})
    
    list_extract_flag_name(args --recurse)
    ans(recurse)

    path_qualify(target_dir)

    glob_ignore(--relative ${args} ${recurse})
    ans(paths)

    pwd()
    ans(pwd)

    foreach(path ${paths})
      path_component(${path} --parent-dir)
      ans(relative_dir)
      file(COPY "${pwd}/${path}" DESTINATION "${target_dir}/${relative_dir}")
     
    endforeach()
    return_ref(paths)
  endfunction()