
function(mime_types_register_default)
  mime_type_register("{
      name:'application/x-gzip',
      description:'',
      extensions:['tgz','gz','tar.gz']
  }")
  mime_type_register("{
      name:'application/zip',
      description:'',
      extensions:['zip']
  }")

  mime_type_register("{
      name:'application/x-serializedcmake',
      description:'',
      extensions:['cmake','scmake']
  }")

  mime_type_register("{
      name:'application/x-7z-compressed',
      description:'',
      extensions:['7z']
  }")

  mime_type_register("{
      name:'text/plain',
      description:'',
      extensions:['txt','asc']
  }")


  mime_type_register("{
      name:'application/x-quickmap',
      description:'CMake Quickmap Object Notation',
      extensions:['qm']
  }")



  mime_type_register("{
      name:'application/json',
      description:'JavaScript Object Notation',
      extensions:['json']
  }")



  mime_type_register("{
      name:'application/x-cmake',
      description:'CMake Script File',
      extensions:['cmake']
  }")



  mime_type_register("{
      name:'application/xml',
      description:'eXtensible Markup Language',
      extensions:['xml']
  }")


endfunction()