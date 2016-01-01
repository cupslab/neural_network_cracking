#!/usr/bin/env sh

mkdir -p build
npm run do-browserify
cd build
java -jar ../compiler/compiler.jar --compilation_level SIMPLE \
     --js worker.js --js_output_file worker.min.js \
     --create_source_map worker.js.source_map \
     --third_party
cd ..
echo "//# sourceMappingURL=/worker.js.source_map" >> build/worker.min.js
java -jar compiler/compiler.jar --compilation_level SIMPLE \
     --js build/nn-client.js --js_output_file build/nn-client.min.js \
     --third_party
