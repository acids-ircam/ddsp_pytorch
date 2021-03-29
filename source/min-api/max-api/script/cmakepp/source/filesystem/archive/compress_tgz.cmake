
function(compress_tgz target_file)
  set(args ${ARGN})
  # target_file file
  path_qualify(target_file)

  # get all files to compress
  glob(${args} --relative)
  ans(paths)

  # compress all files into target_file using paths relative to pwd()
  tar_lean(cvzf "${target_file}" ${paths})
  ans_extract(error)
  return_ans()
endfunction()