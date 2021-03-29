
function(bitbucket_remote_refs user repo ref_type_query ref_name_query )
  set(api_uri "https://bitbucket.org/api/1.0")
  http_get("${api_uri}/repositories/${user}/${repo}/branches-tags" --silent-fail --json)
  ans(refs)
  if(NOT refs)
    return()
  endif()
  set(branches)
  set(tags)
  
  if("${ref_type_query}_" STREQUAL "*_" OR "${ref_type_query}_" STREQUAL "branches_")
    map_tryget(${refs} branches)
    ans(branches)
  endif()
  if("${ref_type_query}_" STREQUAL "*_" OR "${ref_type_query}_" STREQUAL "tags_")
    map_tryget(${refs} tags)
    ans(tags)
  endif()

  set(refs)

  foreach(branch ${branches})
    map_tryget(${branch} name)
    ans(ref)
    map_tryget(${branch} changeset)
    ans(commit)
    set(ref_type "branches")

    if("${ref_name_query}_" STREQUAL "*_" OR "${ref_name_query}_" STREQUAL "${ref}_")
      set(bitbucket_response ${branch})
      map_capture_new(user repo ref_type ref commit bitbucket_response)
      ans_append(refs)
    endif()
  endforeach()
  
  foreach(tag ${tags})
    map_tryget(${tag} name)
    ans(ref)
    map_tryget(${tag} changeset)
    ans(commit)
    set(ref_type "tags")  


    if("${ref_name_query}_" STREQUAL "*_" OR "${ref_name_query}_" STREQUAL "${ref}_")
      set(bitbucket_response ${tag})
      map_capture_new(user repo ref_type ref commit bitbucket_response)
      ans_append(refs)
    endif()

  endforeach()

  return_ref(refs)
endfunction()