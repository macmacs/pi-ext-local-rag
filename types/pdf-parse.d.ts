// pdf-parse ships no types. The inner path bypasses the index.js debug-mode
// side effect (which tries to read a missing test PDF when imported as main).
declare module "pdf-parse/lib/pdf-parse.js" {
  // Takes a Uint8Array, not a Buffer: pdfjs resolves sub-streams against
  // `bytes.buffer` without honouring `byteOffset`, so a pool-backed Buffer
  // makes it read the wrong bytes. See extractText().
  const pdf: (data: Uint8Array) => Promise<{ text: string; numpages?: number }>;
  export default pdf;
}
