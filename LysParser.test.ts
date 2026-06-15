import { describe, it } from 'node:test';
import assert from 'node:assert';
import { LysParser } from './LysParser';

function buildGeometryBlob(options?: {
  headerLength?: number;
  indices?: number[];
  coords?: number[];
}): ArrayBuffer {
  const headerLength = options?.headerLength ?? 20;
  const indices = options?.indices ?? [0, 1, 2];
  const coords = options?.coords ?? [
    0, 0, 0,
    1, 0, 0,
    0, 1, 0,
  ];

  const totalBytes = headerLength + indices.length * 4 + coords.length * 4;
  const buffer = new ArrayBuffer(totalBytes);
  const view = new DataView(buffer);

  // Minimal geometry header
  view.setUint32(0, 1, true); // version
  view.setUint32(4, headerLength, true);
  view.setUint32(8, indices.length, true);
  view.setUint32(12, coords.length, true);
  view.setUint32(16, 0, true);

  let offset = headerLength;
  for (const index of indices) {
    view.setUint32(offset, index, true);
    offset += 4;
  }

  for (const scalar of coords) {
    view.setFloat32(offset, scalar, true);
    offset += 4;
  }

  return buffer;
}

// Builds a stride-48 legacy binary blob: each triangle is [V0, V1, V2, Attr] × 12 bytes.
function buildLegacyStride48Blob(
  triangles: Array<{
    v0: [number, number, number];
    v1: [number, number, number];
    v2: [number, number, number];
    attr: [number, number, number];
  }>,
): ArrayBuffer {
  const STRIDE = 48;
  const buffer = new ArrayBuffer(triangles.length * STRIDE);
  const view = new DataView(buffer);
  const writeVec3 = (off: number, v: [number, number, number]) => {
    view.setFloat32(off,     v[0], true);
    view.setFloat32(off + 4, v[1], true);
    view.setFloat32(off + 8, v[2], true);
  };
  for (let t = 0; t < triangles.length; t++) {
    const { v0, v1, v2, attr } = triangles[t];
    const base = t * STRIDE;
    writeVec3(base,      v0);
    writeVec3(base + 12, v1);
    writeVec3(base + 24, v2);
    writeVec3(base + 36, attr);
  }
  return buffer;
}

// Two non-degenerate triangles: vertex XY ≈ 14–22 mm, attr XY ≈ 0.7 mm.
// After the X↔Z swap: v0 raw(10,20,30) → pos(30,20,10).
const LEGACY_TRIS: Array<{
  v0: [number, number, number];
  v1: [number, number, number];
  v2: [number, number, number];
  attr: [number, number, number];
}> = [
  { v0: [10, 20, 30], v1: [20, 10, 30], v2: [10, 10, 40], attr: [0.7, 0.5, 1.0] },
  { v0: [11, 21, 31], v1: [21, 11, 31], v2: [11, 11, 41], attr: [0.6, 0.4, 0.9] },
];

describe('LysParser.legacyAttrStrideScore', () => {
  it('returns a score ≥ 5 when every 4th triplet has small XY and the others have large XY', () => {
    const buffer = buildLegacyStride48Blob(LEGACY_TRIS);
    const view = new DataView(buffer);
    const score = (LysParser as any).legacyAttrStrideScore(view, 0, buffer.byteLength);
    assert.ok(score >= 5, `Expected score >= 5 for stride-48 pattern, got ${score}`);
  });

  it('returns 0 when all triplets have similar XY magnitudes (no stride pattern)', () => {
    // 8 triplets all with large XY — no attr-stride contrast.
    const buffer = new ArrayBuffer(8 * 12);
    const view = new DataView(buffer);
    for (let i = 0; i < 8; i++) {
      view.setFloat32(i * 12,     20 + i, true);
      view.setFloat32(i * 12 + 4, 15 + i, true);
      view.setFloat32(i * 12 + 8, 10 + i, true);
    }
    const score = (LysParser as any).legacyAttrStrideScore(view, 0, buffer.byteLength);
    assert.strictEqual(score, 0, 'Score should be 0 when no attr-stride XY contrast is present');
  });
});

describe('LysParser.parseLegacyGeometry', () => {
  it('produces the correct vertex count from a stride-48 buffer', () => {
    const buffer = buildLegacyStride48Blob(LEGACY_TRIS);
    const geom = (LysParser as any).parseLegacyGeometry(buffer);
    const posAttr = geom.getAttribute('position');
    assert.strictEqual(posAttr.count, 6, 'Two stride-48 triangles should yield 6 vertices');
  });

  it('swaps raw X↔Z when extracting positions from a stride-48 buffer', () => {
    const buffer = buildLegacyStride48Blob(LEGACY_TRIS);
    const geom = (LysParser as any).parseLegacyGeometry(buffer);
    const positions = geom.getAttribute('position').array as Float32Array;
    // v0 raw: x=10, y=20, z=30 → after X↔Z swap: posX=30, posY=20, posZ=10
    assert.ok(Math.abs(positions[0] - 30) < 1e-4, `Position X should be raw Z (30), got ${positions[0]}`);
    assert.ok(Math.abs(positions[1] - 20) < 1e-4, `Position Y should be raw Y (20), got ${positions[1]}`);
    assert.ok(Math.abs(positions[2] - 10) < 1e-4, `Position Z should be raw X (10), got ${positions[2]}`);
  });

  it('computes vertex normals on the resulting geometry', () => {
    const buffer = buildLegacyStride48Blob(LEGACY_TRIS);
    const geom = (LysParser as any).parseLegacyGeometry(buffer);
    const normalAttr = geom.getAttribute('normal');
    assert.ok(normalAttr, 'Parsed legacy geometry should include a normal attribute');
    assert.ok(normalAttr.count > 0, 'Normal attribute should have at least one entry');
  });

  it('throws when the buffer is too short to contain any vertex data', () => {
    assert.throws(
      () => (LysParser as any).parseLegacyGeometry(new ArrayBuffer(10)),
      (err: unknown) => {
        assert.ok(err instanceof Error, 'Expected Error instance');
        return true;
      },
    );
  });
});

describe('LysParser.parseGeometry', () => {
  it('parses geometry payloads that declare header length larger than 20 bytes', () => {
    const geometry = (LysParser as any).parseGeometry(buildGeometryBlob({ headerLength: 24 }));

    const posAttr = geometry.getAttribute('position');
    assert.ok(posAttr, 'Expected parsed geometry to include position attribute');
    assert.ok(posAttr.count > 0, 'Expected parsed geometry to include vertices');
  });

  it('throws controlled validation errors for malformed payload lengths', () => {
    const malformed = new ArrayBuffer(31);
    const view = new DataView(malformed);

    // Looks like a geometry header, but declares impossible payload lengths.
    view.setUint32(0, 1, true);
    view.setUint32(4, 20, true);
    view.setUint32(8, 4, true);
    view.setUint32(12, 12, true);

    assert.throws(
      () => (LysParser as any).parseGeometry(malformed),
      (err: unknown) => {
        assert.ok(err instanceof Error, 'Expected Error instance');
        assert.ok(!(err instanceof RangeError), 'Expected parser validation error, not typed array RangeError');
        assert.match(err.message, /Geometry payload byte length mismatch/);
        return true;
      },
    );
  });
});
