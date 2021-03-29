# DO NOT EDIT
# This makefile makes sure all linkable targets are
# up-to-date with anything they link to
default:
	echo "Do not invoke directly"

# Rules to remove targets that are older than anything to which they
# link.  This forces Xcode to relink the targets from scratch.  It
# does not seem to check these dependencies itself.
PostBuild.mock_kernel.Debug:
/Users/animation/Documents/Max\ 8/Packages/KlingKlangKlong/tests/libmock_kernel.dylib:
	/bin/rm -f /Users/animation/Documents/Max\ 8/Packages/KlingKlangKlong/tests/libmock_kernel.dylib


PostBuild.mock_kernel.Release:
/Users/animation/Documents/Max\ 8/Packages/KlingKlangKlong/tests/libmock_kernel.dylib:
	/bin/rm -f /Users/animation/Documents/Max\ 8/Packages/KlingKlangKlong/tests/libmock_kernel.dylib


PostBuild.mock_kernel.MinSizeRel:
/Users/animation/Documents/Max\ 8/Packages/KlingKlangKlong/tests/MinSizeRel/libmock_kernel.dylib:
	/bin/rm -f /Users/animation/Documents/Max\ 8/Packages/KlingKlangKlong/tests/MinSizeRel/libmock_kernel.dylib


PostBuild.mock_kernel.RelWithDebInfo:
/Users/animation/Documents/Max\ 8/Packages/KlingKlangKlong/tests/RelWithDebInfo/libmock_kernel.dylib:
	/bin/rm -f /Users/animation/Documents/Max\ 8/Packages/KlingKlangKlong/tests/RelWithDebInfo/libmock_kernel.dylib




# For each target create a dummy ruleso the target does not have to exist
