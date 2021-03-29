## 
##
## invokes the cmakepp project command line interface
function(cmakepp_project_cli)
  #commandline_args_get(--no-script)
  #ans(args)
  set(args ${ARGN})

  list_extract_any_flag(args -g --global)
  ans(global)


  list_extract_any_flag(args -v --verbose)
  ans(verbose)


  if(verbose)

    event_addhandler("on_log_message" "[](entry)message(FORMAT '{entry.function}: {entry.message}')")
    event_addhandler("project_on_opening" "[](proj) message(FORMAT '{event.event_id}: {proj.content_dir}'); message(PUSH)")
    event_addhandler("project_on_opened" "[](proj) message(FORMAT '{event.event_id}')")
    event_addhandler("project_on_loading" "[](proj) message(FORMAT '{event.event_id}'); message(PUSH)")
    event_addhandler("project_on_package_loading" "[](proj pack) message(FORMAT '{event.event_id}: {pack.uri}'); message(PUSH)")
    event_addhandler("project_on_package_loaded" "[](proj pack)  message(POP); message(FORMAT '{event.event_id}: {pack.uri}')")
    event_addhandler("project_on_package_reload" "[](proj pack)   message(FORMAT '{event.event_id}: {pack.uri}')")
    event_addhandler("project_on_package_cycle" "[](proj pack)   message(FORMAT '{event.event_id}: {pack.uri}')")
    event_addhandler("project_on_package_unloading" "[](proj pack) message(FORMAT '{event.event_id}: {pack.uri}'); message(PUSH)")
    event_addhandler("project_on_package_unloaded" "[](proj pack)  message(POP); message(FORMAT '{event.event_id}: {pack.uri}')")
    event_addhandler("project_on_package_materializing" "[](proj pack) message(FORMAT '{event.event_id}: {pack.uri}'); message(PUSH)")
    event_addhandler("project_on_package_materialized" "[](proj pack)  message(POP); message(FORMAT '{event.event_id}: {pack.uri} => {pack.content_dir}')")
    event_addhandler("project_on_package_dematerializing" "[](proj pack) message(FORMAT '{event.event_id}: {pack.uri}'); message(PUSH)")
    event_addhandler("project_on_package_dematerialized" "[](proj pack)  message(POP); message(FORMAT '{event.event_id}: {pack.uri}')")
    event_addhandler("project_on_loaded" "[](proj) message(POP); message(FORMAT '{event.event_id}') ")
    event_addhandler("project_on_closing" "[](proj) message(FORMAT '{event.event_id}'); message(POP)")
    event_addhandler("project_on_closed" "[](proj) message(FORMAT '{event.event_id}: {proj.content_dir}')")
    event_addhandler("project_on_dependency_configuration_changed" "[](proj) message(FORMAT '{event.event_id}: {{ARGN}}')")
    event_addhandler("project_on_dependencies_materializing" "[](proj ) message(FORMAT '{event.event_id}'); message(PUSH)")
    event_addhandler("project_on_dependencies_materialized" "[](proj )  message(POP); message(FORMAT '{event.event_id}')")
    event_addhandler("project_on_package_ready" "[](proj pack)   message(FORMAT '{event.event_id}: {pack.uri}')")
    event_addhandler("project_on_package_unready" "[](proj pack)   message(FORMAT '{event.event_id}: {pack.uri}')")
  endif()


  list_extract_flag(args --save)
  ans(save)


  list_extract_labelled_value(args --project)
  ans(project_dir)

  if(global)
    dir_ensure_exists("~/.cmakepp")
    project_read("~/.cmakepp")
    ans(project)
    assign(project.project_descriptor.is_global = 'true')
  else()
    project_read("${project_dir}")
    ans(project)
  endif()

  list_pop_front(args)
  ans(cmd)

  if(NOT cmd)
    set(cmd run)
  endif()
  
  if("${cmd}" STREQUAL "init")
    list_pop_front(args)
    ans(path)
    project_open("${path}")
    ans(project)
  endif()

  if(NOT project)
    error("no project available")
    return()
  endif()

  map_tryget(${project} project_descriptor)
  ans(project_descriptor)
  map_tryget(${project_descriptor} package_source)
  ans(package_source)
  if(NOT package_source )
    message("no package source found")
    default_package_source()
    ans(package_source)
    map_set(${project_descriptor} package_source ${package_source})
  endif()


  if("${cmd}" STREQUAL "init")
  elseif("${cmd}" STREQUAL "get")

    if("${args}" MATCHES "(.+)\\((.*)\\)$")
      set(path "${CMAKE_MATCH_1}")
      set(call (${CMAKE_MATCH_2}))
    else()
      set(call)
      set(path ${args})
    endif()
    assign(res = "project.${path}" ${call})
  elseif("${cmd}" STREQUAL "set")
    list_pop_front(args)
    ans(path)
    set(call false)
    if("${path}_" STREQUAL "call_")
      list_pop_front(args)
      ans(path)
      set(call true)
    endif()


    if(NOT path)
      error("no path specified")
      return()
    endif()
    if(NOT call)
      assign("!project.${path}" = "'${args}'")
    else()
      list_pop_front(args)
      ans(func)
      assign("!project.${path}" = "${func}"(${args}))
    endif()
    set(save true)
    assign(res = "project.${path}")

  elseif("${cmd}" STREQUAL "run")
    package_handle_invoke_hook("${project}" cmakepp.hooks.run "${project}" "${project}" ${args})
    ans(res)
  else()
    call("project_${cmd}"("${project}" ${args}))
    ans(res)
  endif()

  project_write(${project})
  return_ref(res)

endfunction()