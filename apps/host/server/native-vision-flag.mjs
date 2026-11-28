/** Words an older run config may have left after the Qwen engine's --vision. */
const PICTURE_WORDS = new Set(['on', 'off', 'yes', 'no', 'gpu', 'cpu', 'none']);

/** Gets `args` without --vision and the picture word that follows it (the Qwen engine's --vision stands alone; an older config may
 *  carry a value). Used by native-control.mjs for a load with pictures off. */
export function withoutVisionFlag(args) {
  const list = (Array.isArray(args) ? args : []).map(String), out = [];
  for (let i = 0; i < list.length; i++) {
    if (list[i] === '--vision') { if (i + 1 < list.length && PICTURE_WORDS.has(list[i + 1].toLowerCase())) i++; continue; }
    out.push(list[i]);
  }
  return out;
}
