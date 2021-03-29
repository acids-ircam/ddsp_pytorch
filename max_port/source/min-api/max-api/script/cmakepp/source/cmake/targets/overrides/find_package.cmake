# wrapper for find_package using cps
#find_package(<package> [version] [EXACT] [QUIET]
#             [[REQUIRED|COMPONENTS] [components...]]
#             [NO_POLICY_SCOPE])
macro(find_package)
  set_ans("")
  event_emit(on_find_package ${ARGN})
  if(__ans)
    ## an event returns a cmake package map 
    ## which contains the correct variables
    ## also it contains a hidden field called find_package_return_value
    ## which is the return value for find_package

    scope_import_map("${__ans}")
    map_tryget("${__ans}" find_package_return_value)
  else()  
    _find_package(${ARGN})
    set_ans("")
  endif()
endmacro()