## sample_copy(<sample code:/[0-9][0-9]/> <?target_dir>)
##
## copies the specified sample into the specified directroy
## samples are in the <cmakepp>/samples
  function(sample_copy sample)
      set(args ${ARGN})
      list_pop_back(args)
      ans(target_dir)
      ## copy sample to test dir 
      ## and compile cmakepp to test dir
      cmakepp_config(base_dir)  
      ans(base_dir)
      
      glob("${base_dir}/samples/${sample}*")
      ans(sample_dir )
      cp_dir("${sample_dir}" "${target_dir}")
  endfunction()