"""Check VOP1/VOP2 SDWA translation in tools/startup/gfx2hlsl.py (#1141).

usage: startup_gfx2hlsl_sdwa_check.py

Sub-dword addressing selects a byte or word out of a source, optionally sign-extended, and writes the
low bits of the result back into a byte or word of the destination. The tools must agree on that
meaning, so this drives the translator on gfx1013 assembly and compares every helper result against
tools/pixel_aot.py, which already lowers the same forms.

Pixel AOT refuses a destination select that needs a tracked numeric destination; a register variable
in this translator is always numeric, so only the PAD and PRESERVE forms are compared there.
"""
import os
import sys

here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(here, '..', 'tools'))
sys.path.insert(0, os.path.join(here, '..', 'tools', 'startup'))
import gfx2hlsl as G  # noqa: E402
import pixel_aot as P  # noqa: E402

SELS = ['BYTE_0', 'BYTE_1', 'BYTE_2', 'BYTE_3', 'WORD_0', 'WORD_1', 'DWORD']
failures = []


def check(label, condition):
    print(f"{label}: {'ok' if condition else 'MISMATCH'}")
    if not condition:
        failures.append(label)


# 1. The selection helpers mean the same thing as the pixel AOT ones.
mismatched = [(sel, sext) for sel in SELS for sext in (False, True)
              if G.sdwa_source('x', sel, sext) != P.sdwa_source('x', sel, sext)]
check('source selection parity', not mismatched)
if mismatched:
    print('  differs:', mismatched)
mismatched = [(sel, unused) for sel in SELS for unused in ('UNUSED_PAD', 'UNUSED_PRESERVE')
              if G.sdwa_dest('r', sel, unused, 'alu1') != P.sdwa_dest('r', sel, unused, 'alu1')]
check('destination selection parity', not mismatched)
if mismatched:
    print('  differs:', mismatched)


class Probe(G.Stage):
    """A Stage with no pipeline: enough state for one instruction to be translated."""

    def __init__(self):
        self.lines, self.sym = [], {}
        self.used_s, self.used_v, self.uniform_v = set(), set(), set()
        self.exec_known = True


def translate(assembly):
    stage = Probe()
    stage.one(assembly.split(' ', 1)[0], assembly)
    return ' '.join(stage.lines)


# 2. Byte and word selects reach the emitted HLSL.
emitted = translate('v_add_nc_u32_sdwa v1, v2, v3 dst_sel:BYTE_1 dst_unused:UNUSED_PAD '
                    'src0_sel:BYTE_2 src1_sel:WORD_1')
check('byte destination select emitted', '<< 8u' in emitted and 'v1 =' in emitted)
check('byte source select emitted', '(v2 >> 16u) & 0xffu' in emitted)
check('word source select emitted', '(v3 >> 16u) & 0xffffu' in emitted)
check('operation grouped before the destination mask', '((v2 >> 16u) & 0xffu) + ' in emitted)
print('  emitted:', emitted)

# 3. sext() sign-extends the selected field instead of zero-extending it.
emitted = translate('v_and_b32_sdwa v4, sext(v5), v0 dst_sel:DWORD dst_unused:UNUSED_PAD '
                    'src0_sel:BYTE_0 src1_sel:DWORD')
check('sext emitted as a shift pair', 'asuint(asint(v5 << 24u) >> 24u)' in emitted)
print('  emitted:', emitted)

# 4. UNUSED_PRESERVE keeps the rest of the destination.
emitted = translate('v_or_b32_sdwa v6, v7, v8 dst_sel:WORD_1 dst_unused:UNUSED_PRESERVE '
                    'src0_sel:DWORD src1_sel:DWORD')
check('preserve keeps the destination', 'asuint(v6)' in emitted and '0xffffu' in emitted)
print('  emitted:', emitted)

# 5. A DWORD selection is the identity, so the operation is unchanged.
emitted = translate('v_xor_b32_sdwa v9, v10, v11 dst_sel:DWORD dst_unused:UNUSED_PAD '
                    'src0_sel:DWORD src1_sel:DWORD')
check('dword select is the identity', emitted.replace(' ', '') == 'v9=v10^v11;')
print('  emitted:', emitted)


# 6. Forms with no lowering are refused rather than mistranslated.
def refused(assembly):
    try:
        translate(assembly)
    except G.Unsupported:
        return True
    return False


check('float SDWA refused', refused('v_add_f32_sdwa v1, v2, v3 dst_sel:DWORD dst_unused:UNUSED_PAD '
                                    'src0_sel:DWORD src1_sel:DWORD'))
check('UNUSED_SEXT refused', refused('v_add_nc_u32_sdwa v1, v2, v3 dst_sel:BYTE_0 '
                                     'dst_unused:UNUSED_SEXT src0_sel:DWORD src1_sel:DWORD'))
check('missing source select refused', refused('v_add_nc_u32_sdwa v1, v2, v3 dst_sel:DWORD '
                                               'dst_unused:UNUSED_PAD src0_sel:DWORD'))

if failures:
    print('FAIL')
    for text in failures:
        print('  ' + text)
    sys.exit(1)
print('PASS')