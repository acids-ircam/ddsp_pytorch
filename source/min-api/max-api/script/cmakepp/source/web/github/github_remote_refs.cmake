

## github_remote_refs( <?ref type query>)-> {
##   ref_type: "branches"|"tags"|"commits"
##   ref: <name>
##   commit: <sha>
## }
##
## ref type query ::= "branches"|"tags"|"commits"|"*"
## returns the remote refs for the specified github repository
function(github_remote_refs user repo ref_query)
  set(args ${ARGN})
  list_pop_front(args)
  ans(ref_name_query)

  set(tags)
  set(branches)
  set(commits)
  set(refs)

  if(ref_query AND "${ref_query}" STREQUAL "commits" AND ref_name_query)
      github_api("repos/${user}/${repo}/commits/${ref_name_query}" --exit-code)
      ans(error)
      if(NOT error)
        set(ref ${ref_name_query})
        set(commit ${ref_name_query})
        set(ref_type "commits")
        map_capture_new(ref_type ref commit)
        ans_append(refs)
      endif()
  endif()

  if(ref_query AND "${ref_query}" STREQUAL "*" OR "${ref_query}" STREQUAL "tags")
    github_api("repos/${user}/${repo}/tags" --json --silent-fail)
    ans(tags)
    foreach(tag ${tags})
      assign(ref = tag.name)
      assign(commit = tag.commit.sha)
      set(ref_type "tags")
      map_capture_new(ref_type ref commit)
      ans_append(refs)
    endforeach()
  endif()
  if(ref_query AND "${ref_query}" STREQUAL "*" OR "${ref_query}" STREQUAL "branches")
    github_api("repos/${user}/${repo}/branches" --json --silent-fail)
    ans(branches)

    foreach(branch ${branches})  
      assign(ref = branch.name)
      assign(commit = branch.commit.sha)
      set(ref_type "branches")
      map_capture_new(ref_type ref commit)
      ans_append(refs)
    endforeach()

  endif()

  if(ref_name_query AND NOT "${ref_name_query}" STREQUAL "*")
    set(result)
    foreach(ref ${refs})
      map_tryget(${ref} ref)
      ans(ref_name)
      map_tryget(${ref} commit)
      ans(commit)
      if("${ref_name_query}" STREQUAL "${ref_name}" OR "${ref_name_query}" STREQUAL "${commit}")
        list(APPEND result ${ref})
      endif()
    endforeach()
    set(refs ${result})
  endif()
  return_ref(refs)
endfunction()