((new-object net.webclient).DownloadString('https://raw.github.com/AnotherFoxGuy/cmakepp/master/install.cmake')) |`
out-file -Encoding ascii install.cmake; `
cmake -P install.cmake; `
rm install.cmake;

