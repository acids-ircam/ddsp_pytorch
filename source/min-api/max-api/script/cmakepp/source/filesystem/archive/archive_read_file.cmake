

function(archive_read_file archive file)
  path_qualify(archive)
  mktemp()
  ans(temp_dir)
  uncompress_file("${temp_dir}" "${archive}" "${file}")
  fread("${temp_dir}/${file}")
  ans(content)
  rm("${temp_dir}")
  return_ref(content) 
endfunction()

