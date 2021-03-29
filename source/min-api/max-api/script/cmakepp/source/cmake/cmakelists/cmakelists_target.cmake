## `(<cmakelists> <target:<target name regex>|<cmake target>)-><cmake target> v {target_invocations: <target invocations>}`
##
## tries to find the single target identified by the regex and returns it. 
## 
## ```
## <target> ::= {
##    target_name: <string>
##    target_type: "library"|"executable"|"test"|"custom_target"|...
##    target_source_files
##    target_include_directories
##    target_link_libraries
##    target_compile_definitions
##    target_compile_options
## }
## ```
function(cmakelists_target cmakelists target)
  is_address("${target}")
  ans(is_ref)
  if(is_ref)
    return(${target})
  endif()
  set(target_name ${target})
  cmakelists_targets("${cmakelists}" "${target_name}")
  ans(targets)
  map_values(${targets})
  ans(target)
  list(LENGTH target count)
  if(NOT "${count}" EQUAL 1)
    error("could not find single target (found {count})")
    return()
  endif()
  return_ref(target)
endfunction()
