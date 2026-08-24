// ESM adapter for the self-contained mathjs UMD browser bundle.
//
// The sim modules do `import { create, all } from 'mathjs'`. To vendor mathjs
// fully offline (no CDN), we load its official self-contained UMD build
// (vendor/math.umd.js) via a classic <script> in the page <head> — which sets
// globalThis.math before any module runs — and re-export the named bindings
// the app uses from here. The importmap maps "mathjs" to this file.
const math = globalThis.math;
if (!math) {
  throw new Error(
    'mathjs UMD bundle not loaded: include <script src="vendor/math.umd.js"></script> ' +
    'in the page <head> before the module scripts.'
  );
}
export const create = math.create;
export const all = math.all;
export default math;
