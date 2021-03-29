
# redirects the output of the specified shell to the result value of this function
function(shell_redirect code)
  fwrite_temp("" ".txt")
  ans(tmp_file)
  shell("${code}> \"${tmp_file}\"")
  fread("${tmp_file}")
  ans(res)
  file(REMOVE "${tmp_file}")
  return_ref(res)
endfunction()