// Preloaded into the Pi (Node) worker via NODE_OPTIONS=--import.
//
// Local models served at the edge of a machine's capability (e.g. a dense model
// at high quant on a Mac Studio) can spend many minutes in *prompt processing*
// (prefill) before the first token streams back. Node's global `fetch` is backed
// by undici, whose default `headersTimeout`/`bodyTimeout` is 300s — so a long
// prefill makes undici abort the request mid-generation, which the server logs as
// "Client disconnected. Stopping generation". This disables those read timeouts so
// a single response may take 30+ minutes without the client hanging up.
//
// We resolve undici from the Pi package's own node_modules (PI_FETCH_DISPATCH_BASE
// points at the pi executable). undici writes the dispatcher to the well-known
// global symbol Symbol.for("undici.globalDispatcher.1"), which Node's built-in
// fetch also reads, so this applies whether Pi uses global fetch or undici directly.
import { createRequire } from 'node:module';
import { pathToFileURL } from 'node:url';
import { isAbsolute } from 'node:path';

function resolveBase() {
  // process.argv[1] is the absolute path to Pi's cli.js even during an --import
  // preload, so undici resolves from Pi's own node_modules. The env var and this
  // module's URL are fallbacks. A bare command name (e.g. "pi") is not a valid
  // createRequire base, so only accept absolute paths / URLs.
  const entry = process.argv[1];
  if (entry && isAbsolute(entry)) return pathToFileURL(entry).href;
  const env = process.env.PI_FETCH_DISPATCH_BASE;
  if (env && isAbsolute(env)) return pathToFileURL(env).href;
  return import.meta.url;
}

try {
  const require = createRequire(resolveBase());
  const { setGlobalDispatcher, Agent } = require('undici');
  // 0 disables the timeout in undici. connectTimeout kept modest so a genuinely
  // unreachable host still fails fast; only the post-connect read timeouts (which
  // is what a long prefill trips) are lifted.
  setGlobalDispatcher(
    new Agent({ headersTimeout: 0, bodyTimeout: 0, keepAliveTimeout: 600000 }),
  );
} catch (err) {
  // Best-effort: if undici can't be resolved, leave Node's defaults in place
  // rather than crashing the worker.
  process.stderr.write(`[pi-as-mcp] could not disable fetch read timeouts: ${err}\n`);
}
