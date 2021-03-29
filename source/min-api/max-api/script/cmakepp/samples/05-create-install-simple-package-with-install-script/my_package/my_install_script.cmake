  ## this function generates a cpp main function file 
  ## which when compiled and executed will return 42
  function(my_install_script project_handle package_handle)
    ## here you have access to project_handle and package_handle
    ## which allow you to do anything that you wish for.

    ## in contrast to th on_load callback you do not have access
    ## to exported script files from you package. 

    ## so now lets use the install script to create a library,
    ## compile and install it...

    
    

  endfunction()
