## `([-v])-><any...>`
##
## the comand line interface to cmakelists.  tries to find the CMakelists.txt in current or parent directories
## if init is specified a new cmakelists file is created in the current directory
## *flags*:
##  * 
## *commands*:
##  * `init` saves an initial cmake file at the current location
##  * `target <target name> <target command> | "add" <target name>` target commands:
##    * `add` adds the specified target to the end of the cmakelists file
##    * `sources "append"|"set"|"remove" <glob expression>...` adds appends,sets, removes the source files specified by glob expressions to the specified target
##    * `includes "append"|"set"|"remove" <path>....` adds the specified directories to the target_include_directories of the specified target
##    * `links "append"|"set"|"remove" <target name>...` adds the specified target names to the target_link_libraries of the specified target
##    * `type <target type>` sets the type of the specified target to the specified target type
##    * `rename <target name>` renames the specified target 
## 
## `<target type> ::= "library"|"executable"|"custom_target"|"test"`  
function(cmakelists_cli)
  set(args ${ARGN})
  list_pop_front(args)
  ans(command)

  list_extract_flags(args -v)
  ans(verbose)

  set(handler)
  if(verbose)
    event_addhandler(on_log_message "[](entry) message(FORMAT '{entry.function}: {entry.message}') ")
  endif()
  cmakelists_open("")
  ans(cmakelists)

  if(NOT cmakelists AND "${command}" STREQUAL "init")
    path_parent_dir_name(CMakeLists.txt)
    ans(project_name)
    cmakelists_new("cmake_minimum_required(VERSION ${CMAKE_VERSION})\n\nproject(${project_name})\n\n")
    ans(cmakelists)
  elseif(NOT cmakelists)
    error("no CMakeLists.txt file found in current or parent directories" --function cmakelists_cli)
    return()
  elseif("${command}" STREQUAL "init")
    cmakelists_new("cmake_minimum_required(VERSION ${CMAKE_VERSION})")
    ans(cmakelists)
  endif()

  
  set(save false)
  if("${command}" STREQUAL "init")
    set(save true)
  elseif("${command}" STREQUAL "target")
    list_pop_front(args)
    ans(target_name)

    list_pop_front(args)
    ans(command)

    if("${target_name}" STREQUAL "add")
      set(target_name "${command}")
      set(command add)
    endif()

    if(NOT command)
      cmakelists_targets(${cmakelists} ${target_name})
      ans(result)
    elseif("${command}" STREQUAL "rename")
      list_pop_front(args)
      ans(name)
      cmakelists_target(${cmakelists} "${target_name}")
      ans(target)
      if(NOT target)
          message(FATAL_ERROR FORMAT "no single target found for ${target_name} in {cmakelists.path}")
      endif()
      map_set(${target} target_name "${name}")
      cmakelists_target_update("${cmakelists}" "${target}")
      set(save true)
      set(result ${target})

    elseif("${command}" STREQUAL "type")
      list_pop_front(args)
      ans(type)
      cmakelists_target(${cmakelists} "${target_name}")
      ans(target)
      if(NOT target)
          message(FATAL_ERROR "no single target found for ${target_name}")
      endif()
      map_set(${target} target_type "${type}")
      cmakelists_target_update("${cmakelists}" "${target}")
      set(save true)
      set(result ${target})

    elseif("${command}" STREQUAL add)
      list_extract(args target_type)
      set(save true)
      if(NOT target_type)
          set(target_type library)
      endif()
      map_capture_new(target_name target_type)
      ans(result)
      cmakelists_target_update(${cmakelists} ${result})
    elseif("${command}" STREQUAL "includes")
      cmakelists_target(${cmakelists} "${target_name}")
      ans(target)
      if(NOT target)
          message(FATAL_ERROR "no single target found for ${target_name}")
      endif()

      list_pop_front(args)
      ans(command)

      map_tryget(${target} target_include_directories)
      ans(result)
      if(command)
          set(flag "--${command}")
          set(before ${result})
          cmakelists_paths("${cmakelists}" ${args})
          ans(args)
          list_modify(result ${flag} --remove-duplicates --sort ${args})
          set(save true) 
          map_set(${target} target_include_directories PUBLIC ${result})
          cmakelists_target_update(${cmakelists} ${target})
      endif()
    elseif("${command}" STREQUAL "links")
      cmakelists_target(${cmakelists} "${target_name}")
      ans(target)
      if(NOT target)
          message(FATAL_ERROR "no single target found for ${target_name}")
      endif()

      list_pop_front(args)
      ans(command)

      map_tryget(${target} target_link_libraries)
      ans(result)
      if(command)
          set(flag "--${command}")
          set(before ${result})
          list_modify(result ${flag} --remove-duplicates ${args})

          set(save true)
          map_set(${target} target_link_libraries ${result})
          cmakelists_target_update(${cmakelists} ${target})
                          
      endif()
    elseif("${command}" STREQUAL "sources")

      cmakelists_target(${cmakelists} "${target_name}")
      ans(target)
      if(NOT target)
          message(FATAL_ERROR "no single target found for ${target_name}")
      endif()

      list_pop_front(args )
      ans(command)

      map_tryget(${target} target_source_files)
      ans(result)
      if(command)
          set(flag "--${command}")
          set(before ${result})
          cmakelists_paths(${cmakelists} ${args} --glob)
          ans(args)
          list_modify(result ${flag} --remove-duplicates ${args})

          set(save true)
          map_set(${target} target_source_files ${result})
          cmakelists_target_update(${cmakelists} ${target})
                          
      endif()

    endif()

  endif()

  if(save)

    cmakelists_close(${cmakelists})
  endif()
  return_ref(result)
endfunction()




function(cmakelists_target_modify cmakelists target target_property)
  set(args ${ARGN})
  cmakelists_target("${cmakelists}" "${target}")
  ans(target)
  if(NOT target)
    return()
  endif()
  
  list_pop_front(args)
  ans(command)

  map_tryget(${target} "${target_property}")
  ans(result)
  
  if(command)
    set(flag "--${command}")
    list_modify(result ${flag} --remove-duplicates ${args})
    print_vars(result command args target.target_name target_property)
    map_set(${target} "${target_property}" ${result})
    cmakelists_target_update(${cmakelists} ${target})
  endif()        
  return_ref(result)
endfunction()