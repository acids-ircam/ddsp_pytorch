## returns the number of milliseconds since epoch
function(millis)

  compile_tool(millis "
    #include <iostream>
    #include <chrono>
    int main(int argc, const char ** argv){
     //std::cout << \"message(whatup)\"<<std::endl;
     //std::cout << \"obj(\\\"{id:'1'}\\\")\" <<std::endl;
     auto now = std::chrono::system_clock::now();
     auto duration = now.time_since_epoch();
     auto millis = std::chrono::duration_cast<std::chrono::milliseconds>(duration).count();
     std::cout<< \"set_ans(\" << millis << \")\";
     return 0;
    }"
    )
  millis(${ARGN})
  return_ans()
endfunction()

