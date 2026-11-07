/** Gets GPU usage from hardware telemetry, falling back to the Memory panel's readings.
 * A physical GPU may have several engine partitions; its capacity is counted once. */
export function getVramReading(hardware, memories = []) {
  if (hardware.gpu_mem_used != null)
    return {used: hardware.gpu_mem_used, total: hardware.gpu_mem_total, source: "telemetry",
      gpus: hardware.gpus || []};
  const groups = new Map();
  for (const memory of memories) {
    const match = /^gpu(\d+)(?:\.\d+)?$/.exec(memory.id);
    if (!match)
      continue;
    const index = Number(match[1]), group = groups.get(index) || [];
    group.push(memory);
    groups.set(index, group);
  }
  const cards = [...groups].map(([index, group]) => {
    const primary = group.find(memory => memory.id === `gpu${index}`) || group[0];
    const values = group.flatMap(memory => Object.values(memory.uses || {}));
    const allocated = values.length ? values.reduce((sum, bytes) => sum + bytes, 0) : null;
    return {index, total: primary.total, free: primary.free,
      allocated: primary.in_share ?? allocated};
  });
  const wholeDevice = cards.length > 0 && cards.every(card =>
    card.total > 0 && card.free != null && card.free >= 0 && card.free <= card.total);
  const gpus = cards.map(card => ({index: card.index, mem_total: card.total,
    mem_used: wholeDevice ? card.total - card.free : card.allocated}));
  const complete = gpus.length > 0 && gpus.every(card => card.mem_used != null);
  const total = gpus.length > 0 && gpus.every(card => card.mem_total > 0) ?
    gpus.reduce((sum, card) => sum + card.mem_total, 0) : null;
  return {used: complete ? gpus.reduce((sum, card) => sum + card.mem_used, 0) : null, total,
    source: wholeDevice ? "memory" : "engine", gpus};
}
