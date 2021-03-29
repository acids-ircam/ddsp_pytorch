
function(file_anchor_require_dir anchorName)
  file_find_anchor(.packages)
  ans(packageAnchor)

  if("${packageAnchor}_" STREQUAL "_")
    path(".packages")
    ans(packageAnchor)
    mkdir("${packageAnchor}")
  endif()

  return_ref(packageAnchor) 
endfunction()
