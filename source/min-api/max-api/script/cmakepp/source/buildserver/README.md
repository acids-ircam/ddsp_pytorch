# Buildserver

This is exploratory documention on creating a platform independant multiplatform build server. 



`buildserver(<start options>)`

```
<start options> ::= {
  containers:
    - container id
    - container id

  environment
    -
    - 

  # some kind of build matrix

  # build instructions
  from travis
  -before_install
  -install
  -script
  -after_script
  -after_success
  -after_failure
  etc

}

build configuration ::=
{
  container
  environment
  execute script{
    setup environment
    download repository
    before install
    install
    after_install
    script{
      success : aftersuccess 
      failure : after_failure
    }
    after_script

  }
}

```

defualt process

kick off build (triggered by anything)

configurations = convert matrix to configurations

foreach configuration in configurations
  container start configuration.container -> wait handle
endforeach

wait for all wait handles

retrieve results 




`<build container> := <virtual machine slave>|<docker instance slave>|<local machine slave>|<ssh slave>|...`


build container:  
* container start -> identifier -> result identifier
* execute code on container (via PUT and GET or ssh) 
* retrieve results(identifier) -> result of build
* delete cotnainer

