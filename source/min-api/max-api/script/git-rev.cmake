# Copyright 2018 The Max-API Authors. All rights reserved.
# Use of this source code is governed by the MIT License found in the License.md file.

include("${CMAKE_CURRENT_LIST_DIR}/GetGitRevisionDescription.cmake")

set(HASH "error")
get_git_head_revision(REFSPEC HASH --always --tags)
git_describe(GIT_TAG --abbrev=0 --tags)

if(NOT ${HASH} STREQUAL "error" AND NOT ${HASH} STREQUAL "GIT-NOTFOUND" AND NOT ${GIT_TAG} STREQUAL "HEAD-HASH-NOTFOUND")
string(SUBSTRING ${HASH} 0 7 GIT_SHA_SHORT)
message("building on Git rev : " ${GIT_SHA_SHORT})
message("Git tag : " ${GIT_TAG})

string(REPLACE "v" "" GIT_VERSION_TAG "${GIT_TAG}")

string(LENGTH "${GIT_VERSION_TAG}" taglen)
#message("Git tag length : " ${taglen})
if (taglen GREATER 1)
  string(REPLACE "." ";" GIT_TAG_LIST ${GIT_VERSION_TAG}) # make a list from the tag string
  list(LENGTH GIT_TAG_LIST len)
  if (len GREATER 0)
          list(GET GIT_TAG_LIST 0 GIT_VERSION_MAJ)
  endif (len GREATER 0)
  if (len GREATER 1)
          list(GET GIT_TAG_LIST 1 GIT_VERSION_MIN)
  endif (len GREATER 1)
  if (len GREATER 2)
          list(GET GIT_TAG_LIST 2 GIT_VERSION_SUB)
  endif (len GREATER 2)
  if (len GREATER 3)
          list(GET GIT_TAG_LIST 3 GIT_VERSION_MOD_LONG)
  endif (len GREATER 3)
  list(LENGTH GIT_VERSION_MOD_LONG len2)
  if (len2 GREATER 0)
          string(REPLACE "-" ";" GIT_VERSION_MOD_LIST ${GIT_VERSION_MOD_LONG})
          list(GET GIT_VERSION_MOD_LIST 0 GIT_VERSION_MOD)
  endif (len2 GREATER 0)
endif (taglen GREATER 1)
else()
  message("using default version : 1.0.0")
  set(GIT_VERSION_MAJ 1)
  set(GIT_VERSION_MIN 0)
  set(GIT_VERSION_SUB 0)
  set(GIT_VERSION_MOD_LONG 0)
endif()

