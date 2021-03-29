# returns a list of ref maps containing the fields 
# name type and revision
function(git_remote_refs uri)
  git_uri("${uri}")
  ans(uri)

  git_lean(ls-remote ${uri})
  ans_extract(error)
  ans(stdout)

  if(error)
    return()
  endif()

  string_split( "${stdout}" "\n")
  ans(lines)
  set(res)
  foreach(line ${lines})
    string(STRIP "${line}" line)

    # match
    if("${line}" MATCHES "^([0-9a-fA-F]*)\t(.*)$")
      string(REGEX REPLACE "^([0-9a-fA-F]*)\t(.*)$" "\\1;\\2" parts "${line}")
      list_extract(parts revision ref)
      git_ref_parse("${ref}")
      ans(ref_map)
      
      map_set("${ref_map}" uri "${uri}")
      if(ref_map)
        map_set(${ref_map} revision ${revision})
        set(res ${res} ${ref_map})
        #address_print(${ref_map})
      endif()
    endif()
  endforeach()   
  return_ref(res)
endfunction()