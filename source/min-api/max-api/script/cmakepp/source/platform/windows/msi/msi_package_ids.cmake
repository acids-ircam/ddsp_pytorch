
## `()-><ms guid>`
##
## queries the windows registry for packages installed with msi
## returns their ids (which are microsoft guid)
function(msi_package_ids)
  reg_lean(query "HKLM\\SOFTWARE\\Microsoft\\Windows\\CurrentVersion\\Uninstall")
  ans_extract(error)
  ans(entries)
  if(error)
    message(FATAL_ERROR "could not query register for msi installed packages")
  endif()
  string(REPLACE "\n" ";" entries "${entries}")
  regex_common()

  set(installations)
  foreach(entry ${entries})
    if("${entry}" MATCHES "\\\\(${regex_guid_ms})$")
      list(APPEND installations "${CMAKE_MATCH_1}")
    endif()
  endforeach()
  return_ref(installations)
endfunction()