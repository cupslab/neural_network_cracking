#!/usr/bin/env sh

mkdir -p build
npm run do-browserify
java -jar compiler/compiler.jar \
     --js build/worker.js --js_output_file build/worker.min.js
java -jar compiler/compiler.jar \
     --js build/nn-client.js --js_output_file build/nn-client.min.js
