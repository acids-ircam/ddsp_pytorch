
    function(bitbucket_read_file user repo ref path)
      set(raw_uri "https://bitbucket.org/${user}/${repo}/raw/${ref}/${path}")
      http_get("${raw_uri}" "" --silent-fail)
      return_ans()
    endfunction()