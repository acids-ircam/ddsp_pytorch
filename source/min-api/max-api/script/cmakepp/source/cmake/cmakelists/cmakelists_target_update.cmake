## `(<cmakelists> <cmake target>)-><bool>`
## 
## updates the cmakelists tokens to reflect changes in the target
## @todo extrac functions
## 
function(cmakelists_target_update cmakelists target)
  cmakelists_target("${cmakelists}" "${target}")
  ans(target)
  if(NOT target)
    return(false)
  endif()

  map_tryget(${target} target_invocations)
  ans(target_invocations)

  map_tryget(${target} target_name)
  ans(target_name)
  if(NOT target_name)
    error("target name not specified" --function cmakelists_target_update)
    return(false)
  endif()
  
  map_tryget(${target} target_type)
  ans(target_type)

  ## target does not exist. create
  if(NOT target_invocations)
    log("adding target ${target_name} (${target_type}) to end of cmakelists file" --trace --function cmakelists_target_update)
    ## find insertion point
    map_tryget(${cmakelists} range)
    ans_extract(begin end)

    cmake_token_range_insert("${end}" "\nadd_${target_type}(${target_name})\n")
    cmakelists_target("${cmakelists}" "${target_name}")
    ans(new_target)


    map_defaults("${target}" "${new_target}")
    map_tryget(${new_target} target_invocations)
    ans(target_invocations)
    map_set_hidden(${target} target_invocations ${target_invocations})
    
  endif()

  ## sets the target type  
  map_tryget(${target_invocations} target_source_files)
  ans(target_definition_invocation)
  map_tryget(${target_definition_invocation} invocation_token)
  ans(target_definition_invocation_token)
  map_set("${target_definition_invocation_token}" value "add_${target_type}")


  map_tryget(${target_definition_invocation_token} column)
  ans(insertion_column)
  string_repeat(" " ${insertion_column})
  ans(indentation)



  foreach(current_property 
    target_source_files 
    target_link_libraries 
    target_include_directories 
    target_compile_options
    target_compile_definitions
    )


    map_tryget(${target} "${current_property}")
    ans(values)
    map_tryget(${target_invocations} "${current_property}")
    ans(invocation)
    list(LENGTH values has_values)
    if(has_values)
      if(NOT invocation)
        log("adding ${current_property} for ${target_name}" --trace --function cmakelists_target_update)
        map_tryget(${target_definition_invocation} arguments_end_token)
        ans(insertion_point)
        cmake_token_range_filter("${insertion_point}" type MATCHES "(new_line)|(eof)")
        ans_extract(insertion_point)         
        cmake_token_range_insert("${insertion_point}" "\n${indentation}${current_property}()")
        ans_extract(invocation_token)
        cmake_token_range_filter("${invocation_token}" type STREQUAL "command_invocation")
        ans_extract(invocation_token)
        map_set("${invocation_token}" "column" ${insertion_column})
      else()
        log("updating ${current_property} for ${target_name} to '${values}'" --trace --function cmakelists_target_update)
        map_tryget(${invocation} invocation_token)
        ans(invocation_token)
      endif()
      cmake_invocation_token_set_arguments("${invocation_token}" "${target_name}" ${values})

    elseif(invocation AND NOT "${current_property}"  STREQUAL "target_source_files")
      log("removing ${current_property} for ${target_name}" --trace --function cmakelists_target_update)
      ## remove invocation
      map_remove("${target_invocations}" "${current_property}")
      cmake_invocation_remove("${invocation}")
    endif()

  endforeach()    
  return(true)
endfunction()