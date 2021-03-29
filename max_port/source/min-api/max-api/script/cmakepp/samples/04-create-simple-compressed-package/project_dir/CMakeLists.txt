    

    cmake_minimum_required(VERSION 2.8.12)

    ## be sure to have the compiled cmakepp in the project dir
    include("cmakepp.cmake")

    ## opens the project in the current source dir
    ## this loads all installed packages 
    ## also giving you access to exported cmake functions 
    project_read(--force)
    

    ## the following code creates a simple cmake classic project 
    ## with a single executable
    project(sample03)

    my_sample_function()
    ans(generated_file)

    # and create an executable form it
    add_executable(myexe ${generated_file})