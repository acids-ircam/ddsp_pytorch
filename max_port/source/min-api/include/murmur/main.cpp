
#include "Murmur3.h"
  


int main()
{

  const uint32_t hash = std::integral_constant<uint32_t, Murmur3_32("some_string_to_hash", 0xAED123FD)>::value;

	
  std::cerr << hash << std::endl;
	
  assert(hash == 4291478129);
	
	
	
	constexpr uint32_t i = Murmur3_32("foo");
	std::cerr << i << std::endl;

}

