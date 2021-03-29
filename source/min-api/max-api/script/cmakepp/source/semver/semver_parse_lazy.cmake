

function(semver_parse_lazy version_string)
  if(NOT version_string)
    return()
  endif()
  string_take_regex(version_string "v")


  map_new()
  ans(version)
  map_set(${version} string "${version_string}")


  set(version_number_regex "[0-9]+")
  set(identifier_regex "[0-9a-zA-Z]+")
  set(version_numbers_regex "(${version_number_regex}(\\.${version_number_regex}(\\.${version_number_regex})?)?)")

  # checks if version is of ()-()+() structure and only contains valid characters
  set(version_elements_regex "([0-9\\.]*(-[a-zA-Z0-9\\.-]*)?(\\+[a-zA-Z0-9\\.-]*)?)")
  set(valid)
  string(REGEX MATCH "^${version_elements_regex}$" valid "${version_string}")
  if(NOT valid)
    return()
  endif()
  # split into version string and prelrelease metadata
  string_split_at_first(version_numbers prerelease_and_metadata "${version_string}" "-")
  string_split_at_first(prerelease metadata "${prerelease_and_metadata}" "+")
  # parse version numbers
  if(version_numbers)
    string(REGEX MATCH "^${version_numbers_regex}$" valid "${version_numbers}")
    if(NOT valid)
      return()
    endif()
    string(REPLACE "." ";" version_numbers "${version_numbers}")
    string(REPLACE "." ";" metadatas "${metadata}")
    string(REPLACE "." ";" tags "${prerelease}")
    list_extract(version_numbers major minor patch)
    map_set(${version} numbers "${version_numbers}")
    map_set(${version} major "${major}")
    map_set(${version} minor "${minor}")
    map_set(${version} patch "${patch}")
    #nav("version.numbers" "${version_numbers}")
    #nav("version.major" "${major}")
    #nav("version.minor" "${minor}")
    #nav("version.patch" "${patch}")
  endif()

  #nav("version.prerelease" "${prerelease}")
  #nav("version.metadata" "${metadata}")
  #nav("version.metadatas" "${metadatas}")
  #nav("version.tags" "${tags}")
  map_set(${version} prerelease "${prerelease}")
  map_set(${version} metadata "${metadata}")
  map_set(${version} metadatas "${metadatas}")
  map_set(${version} tags "${tags}")

  return(${version})
endfunction()
