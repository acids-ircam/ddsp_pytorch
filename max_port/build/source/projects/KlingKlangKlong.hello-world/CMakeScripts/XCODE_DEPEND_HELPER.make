# DO NOT EDIT
# This makefile makes sure all linkable targets are
# up-to-date with anything they link to
default:
	echo "Do not invoke directly"

# Rules to remove targets that are older than anything to which they
# link.  This forces Xcode to relink the targets from scratch.  It
# does not seem to check these dependencies itself.
PostBuild.KlingKlangKlong.hello-world.Debug:
/Users/animation/Documents/Max\ 8/Packages/KlingKlangKlong/externals/KlingKlangKlong.hello-world.mxo/Contents/MacOS/KlingKlangKlong.hello-world:
	/bin/rm -f /Users/animation/Documents/Max\ 8/Packages/KlingKlangKlong/externals/KlingKlangKlong.hello-world.mxo/Contents/MacOS/KlingKlangKlong.hello-world


PostBuild.KlingKlangKlong.hello-world_test.Debug:
PostBuild.mock_kernel.Debug: /Users/animation/Documents/Max\ 8/Packages/KlingKlangKlong/tests/KlingKlangKlong.hello-world_test
/Users/animation/Documents/Max\ 8/Packages/KlingKlangKlong/tests/KlingKlangKlong.hello-world_test:\
	/Users/animation/Documents/Max\ 8/Packages/KlingKlangKlong/tests/libmock_kernel.dylib
	/bin/rm -f /Users/animation/Documents/Max\ 8/Packages/KlingKlangKlong/tests/KlingKlangKlong.hello-world_test


PostBuild.KlingKlangKlong.hello-world.Release:
/Users/animation/Documents/Max\ 8/Packages/KlingKlangKlong/externals/KlingKlangKlong.hello-world.mxo/Contents/MacOS/KlingKlangKlong.hello-world:
	/bin/rm -f /Users/animation/Documents/Max\ 8/Packages/KlingKlangKlong/externals/KlingKlangKlong.hello-world.mxo/Contents/MacOS/KlingKlangKlong.hello-world


PostBuild.KlingKlangKlong.hello-world_test.Release:
PostBuild.mock_kernel.Release: /Users/animation/Documents/Max\ 8/Packages/KlingKlangKlong/tests/KlingKlangKlong.hello-world_test
/Users/animation/Documents/Max\ 8/Packages/KlingKlangKlong/tests/KlingKlangKlong.hello-world_test:\
	/Users/animation/Documents/Max\ 8/Packages/KlingKlangKlong/tests/libmock_kernel.dylib
	/bin/rm -f /Users/animation/Documents/Max\ 8/Packages/KlingKlangKlong/tests/KlingKlangKlong.hello-world_test


PostBuild.KlingKlangKlong.hello-world.MinSizeRel:
/Users/animation/Documents/Max\ 8/Packages/KlingKlangKlong/externals/MinSizeRel/KlingKlangKlong.hello-world.mxo/Contents/MacOS/KlingKlangKlong.hello-world:
	/bin/rm -f /Users/animation/Documents/Max\ 8/Packages/KlingKlangKlong/externals/MinSizeRel/KlingKlangKlong.hello-world.mxo/Contents/MacOS/KlingKlangKlong.hello-world


PostBuild.KlingKlangKlong.hello-world_test.MinSizeRel:
PostBuild.mock_kernel.MinSizeRel: /Users/animation/Documents/Max\ 8/Packages/KlingKlangKlong/tests/MinSizeRel/KlingKlangKlong.hello-world_test
/Users/animation/Documents/Max\ 8/Packages/KlingKlangKlong/tests/MinSizeRel/KlingKlangKlong.hello-world_test:\
	/Users/animation/Documents/Max\ 8/Packages/KlingKlangKlong/tests/MinSizeRel/libmock_kernel.dylib
	/bin/rm -f /Users/animation/Documents/Max\ 8/Packages/KlingKlangKlong/tests/MinSizeRel/KlingKlangKlong.hello-world_test


PostBuild.KlingKlangKlong.hello-world.RelWithDebInfo:
/Users/animation/Documents/Max\ 8/Packages/KlingKlangKlong/externals/RelWithDebInfo/KlingKlangKlong.hello-world.mxo/Contents/MacOS/KlingKlangKlong.hello-world:
	/bin/rm -f /Users/animation/Documents/Max\ 8/Packages/KlingKlangKlong/externals/RelWithDebInfo/KlingKlangKlong.hello-world.mxo/Contents/MacOS/KlingKlangKlong.hello-world


PostBuild.KlingKlangKlong.hello-world_test.RelWithDebInfo:
PostBuild.mock_kernel.RelWithDebInfo: /Users/animation/Documents/Max\ 8/Packages/KlingKlangKlong/tests/RelWithDebInfo/KlingKlangKlong.hello-world_test
/Users/animation/Documents/Max\ 8/Packages/KlingKlangKlong/tests/RelWithDebInfo/KlingKlangKlong.hello-world_test:\
	/Users/animation/Documents/Max\ 8/Packages/KlingKlangKlong/tests/RelWithDebInfo/libmock_kernel.dylib
	/bin/rm -f /Users/animation/Documents/Max\ 8/Packages/KlingKlangKlong/tests/RelWithDebInfo/KlingKlangKlong.hello-world_test




# For each target create a dummy ruleso the target does not have to exist
/Users/animation/Documents/Max\ 8/Packages/KlingKlangKlong/tests/MinSizeRel/libmock_kernel.dylib:
/Users/animation/Documents/Max\ 8/Packages/KlingKlangKlong/tests/RelWithDebInfo/libmock_kernel.dylib:
/Users/animation/Documents/Max\ 8/Packages/KlingKlangKlong/tests/libmock_kernel.dylib:
