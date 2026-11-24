"""gfx1013 machine code -> HLSL for a vertex+pixel pipeline (startup tools; see docs/startup-runtime.md).

Model: one lane. Registers are uint variables; exec/vcc/mask registers hold the lane's bit in bit 0.
Scalar registers are also tracked symbolically at translation time so that pointer and descriptor
values (tables, vertex-buffer table, root buffers) turn into HLSL resources instead of arithmetic.
The output is HLSL for both stages plus a manifest that tells the runtime what to bind where.
"""
import os
import re
import struct
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import disasm_rdna2 as D  # noqa: E402


class Unsupported(Exception):
    pass


PREFETCH = bytes.fromhex('0300a0bf')

# SDWA (sub-dword addressing) selections, with the same meaning as tools/pixel_aot.py: a source
# selection is zero-extended, or sign-extended under sext(); a destination selection writes the low
# bits of the result into that field, UNUSED_PAD zeroes the others and UNUSED_PRESERVE keeps them.
SDWA_SEL = {'BYTE_0': (0, 8), 'BYTE_1': (8, 8), 'BYTE_2': (16, 8), 'BYTE_3': (24, 8),
            'WORD_0': (0, 16), 'WORD_1': (16, 16), 'DWORD': (0, 32)}


def sdwa_source(word, sel, sext):
    """The uint expression a source selection reads out of a 32-bit register."""
    offset, width = SDWA_SEL[sel]
    if width == 32:
        return word
    if sext:
        return 'asuint(asint(%s << %du) >> %du)' % (word, 32 - offset - width, 32 - width)
    return '((%s >> %du) & %#xu)' % (word, offset, (1 << width) - 1)


def sdwa_dest(result, sel, unused, old):
    """The uint expression a destination selection writes back into a 32-bit register."""
    offset, width = SDWA_SEL[sel]
    if width == 32:
        return result
    if unused == 'UNUSED_SEXT':
        raise Unsupported('SDWA UNUSED_SEXT')
    field = '((%s & %#xu) << %du)' % (result, (1 << width) - 1, offset)
    if unused == 'UNUSED_PAD':
        return field
    if unused != 'UNUSED_PRESERVE':
        raise Unsupported('SDWA dst_unused %s' % unused)
    keep = ~(((1 << width) - 1) << offset) & 0xffffffff
    return '((asuint(%s) & %#xu) | %s)' % (old, keep, field)


def strip_header(code):
    at = code.find(PREFETCH, 0, 0x80)
    return code[at:] if at >= 0 and at % 4 == 0 else code


def decode(code):
    out, off = [], 0
    while off + 4 <= len(code):
        try:
            n, name, asm, _ = D.decode_instruction(code, off)
        except Exception as error:
            raise Unsupported('undecodable word at +0x%x: %s' % (off, error))
        out.append((off, n, name, asm))
        off += n
    last = max((i for i, x in enumerate(out) if x[2] == 's_endpgm'), default=len(out) - 1)
    return out[:last + 1]


def split_top(text, sep=','):
    parts, depth, cur = [], 0, ''
    for ch in text:
        if ch in '[(':
            depth += 1
        elif ch in '])':
            depth -= 1
        if ch == sep and depth == 0:
            parts.append(cur.strip())
            cur = ''
        else:
            cur += ch
    if cur.strip():
        parts.append(cur.strip())
    return parts


def parse(asm):
    """-> (mnemonic, [operands], [modifier tokens])"""
    name, _, rest = asm.partition(' ')
    ops = split_top(rest)
    mods = []
    if ops:
        tail = split_top(ops[-1], ' ')
        ops[-1], mods = tail[0], tail[1:]
    return name, ops, mods


FLOATS = {'0.5': 0.5, '-0.5': -0.5, '1.0': 1.0, '-1.0': -1.0, '2.0': 2.0, '-2.0': -2.0, '4.0': 4.0, '-4.0': -4.0,
          '0.15915494': 0.15915494}


def fbits(x):
    return struct.unpack('<I', struct.pack('<f', x))[0]


def reg_range(tok):
    """'s[8:11]' -> ('s', 8, 11); 's8' -> ('s', 8, 8); 'v[1:3]' -> ('v', 1, 3); 'exec'/'vcc' pairs too."""
    m = re.fullmatch(r'([sv])\[(\d+):(\d+)\]', tok)
    if m:
        return m.group(1), int(m.group(2)), int(m.group(3))
    m = re.fullmatch(r'([sv])(\d+)', tok)
    if m:
        return m.group(1), int(m.group(2)), int(m.group(2))
    return None


SAMPLER_CLAMP = {0: 'WRAP', 1: 'MIRROR', 2: 'CLAMP', 3: 'MIRROR_ONCE', 4: 'CLAMP', 5: 'MIRROR_ONCE', 6: 'BORDER', 7: 'BORDER'}


class Stage:
    """Translates one stage. Shares the resource and entry-constant registries of its pipeline."""

    def __init__(self, pipe, kind, code):
        self.pipe, self.kind = pipe, kind
        self.ins = decode(strip_header(code))
        self.lines = []           # body statements (per block)
        self.sym = {}             # scalar register -> symbolic value
        self.used_s, self.used_v = set(), set()
        self.exec_known = True    # exec still equals the entry mask (the lane is active)
        self.exports = {}         # 'pos0' / 'paramN' / 'mrtN' -> [4 expressions]
        self.attrs = {}           # pixel: attr index -> 'linear' | 'flat'
        self.uses_pos = False
        self.masks = set()        # scalar registers currently holding lane masks
        self.setup_entry()
        self.translate()

    # ---- entry state ---------------------------------------------------------------------
    def setup_entry(self):
        root = self.pipe.root
        reg = 8
        for index, p in enumerate(root['params']):
            if p['type'] == 0:
                self.sym[reg] = ('ptr', ('table', index))
                reg += 1
            elif p['type'] == 1:
                for i in range(p['count']):
                    self.sym[reg] = ('ud', self.pipe.ud(('rootconst', index, i)))
                    reg += 1
            else:
                self.sym[reg] = ('ptr', ('root', index))
                self.sym[reg + 1] = ('ptrhi', ('root', index))
                reg += 2
        if root['flags'] & 0x400:
            self.sym[reg] = ('ptr', ('resheap',))
            reg += 1
        if root['flags'] & 0x800:
            self.sym[reg] = ('ptr', ('smpheap',))
            reg += 1
        if self.kind == 'vs':
            self.sym[0] = ('ptr', ('vbtab',))
            self.sym[1] = ('ptrhi', ('vbtab',))
            self.sym[2] = ('ud', self.pipe.ud(('basevertex',)))
            self.sym[3] = ('ud', self.pipe.ud(('startinstance',)))
        else:
            self.sym[reg] = ('const', 0)  # primitive mask: only ever moved to m0

    # ---- operand helpers ---------------------------------------------------------------------
    def S(self, n):
        """Runtime expression of scalar register n (uint)."""
        s = self.sym.get(n)
        if s:
            if s[0] == 'const':
                return '0x%08xu' % (s[1] & 0xffffffff)
            if s[0] == 'ud':
                return 'ud%d' % s[1]
            if s[0] in ('execcopy', 'fixed_buffer_word'):
                self.used_s.add(n)
                return 's%d' % n
            raise Unsupported('numeric use of s%d holding %s' % (n, s[0]))
        self.used_s.add(n)
        return 's%d' % n

    def u(self, tok):
        """uint expression for a 32-bit operand."""
        if tok in ('vcc_lo', 'vcc'):
            return 'vcc'
        if tok in ('exec_lo', 'exec'):
            return 'exec'
        if tok == 'm0':
            return 'm0'
        r = reg_range(tok)
        if r:
            if r[0] == 'v':
                self.used_v.add(r[1])
                return 'v%d' % r[1]
            return self.S(r[1])
        if tok in FLOATS:
            return '0x%08xu' % fbits(FLOATS[tok])
        try:
            return '0x%08xu' % (int(tok, 0) & 0xffffffff)
        except ValueError:
            raise Unsupported('operand %r' % tok)

    def f(self, tok):
        """float expression, with the inline neg/abs operand modifiers."""
        neg = False
        if tok.startswith('-') and not re.match(r'-[\d.]', tok):
            neg, tok = True, tok[1:]
        absolute = tok.startswith('|') and tok.endswith('|')
        if absolute:
            tok = tok[1:-1]
        if tok in FLOATS:
            text = repr(FLOATS[tok])
        elif tok == '-0':
            text = '0.0'
        else:
            text = 'F(%s)' % self.u(tok)
        if absolute:
            text = 'abs(%s)' % text
        return '(-%s)' % text if neg else text

    def is_const(self, n):
        s = self.sym.get(n)
        return s[1] if s and s[0] == 'const' else None

    def emit(self, text):
        self.lines.append(text)

    def set_v(self, n, expr):
        self.uniform_v.discard(n)
        self.used_v.add(n)
        self.emit('v%d = %s;' % (n, expr) if self.exec_known else 'if (exec & 1) v%d = %s;' % (n, expr))

    def set_vf(self, n, expr, mods=()):
        for m in mods:
            if m == 'clamp':
                expr = 'saturate(%s)' % expr
            elif m in ('mul:2', 'mul:4'):
                expr = '(%s) * %s.0' % (expr, m[-1])
            elif m == 'div:2':
                expr = '(%s) * 0.5' % expr
            elif m.startswith('op_sel'):
                pass
            else:
                raise Unsupported('modifier %s' % m)
        self.set_v(n, 'U(%s)' % expr)

    def set_s(self, n, expr, sym=None, scc=None):
        """Assign scalar register n at run time and/or symbolically."""
        if sym is not None:
            self.sym[n] = sym
            if sym[0] != 'const':
                return
        else:
            self.sym.pop(n, None)
        self.used_s.add(n)
        self.emit('s%d = %s;' % (n, expr))
        if scc:
            self.emit('scc = %s;' % scc.replace('$', 's%d' % n))

    def descriptor(self, tok):
        """Resource key for a descriptor operand like s[8:11]: (source, byte offset or None)."""
        r = reg_range(tok)
        s = self.sym.get(r[1]) if r else None
        if s and s[0] == 'desc':
            return s[1], s[2]
        if s and s[0] == 'ptr':
            return s[1], None
        raise Unsupported('descriptor operand %s holds %s' % (tok, s))

    def heap_sampler(self, tok):
        """'qh[...]' when the four sampler words of tok came from one run-time sampler heap load."""
        r = reg_range(tok)
        if not r or r[2] - r[1] != 3:
            return None
        first = self.sym.get(r[1])
        if not first or first[0] != 'smpheap':
            return None
        if any(self.sym.get(r[1] + i) != ('smpheap', first[1], i) for i in range(4)):
            raise Unsupported('sampler operand %s mixes sampler heap words' % tok)
        self.pipe.uses_sampler_heap = True
        # Sampler descriptors are 16 bytes; x2w_sampler_entry() maps the byte offset to its entry and
        # sends unaligned or out-of-range offsets to the fallback entry.
        return 'qh[x2w_sampler_entry(s%d)]' % first[1]

    def const_words(self, tok):
        r = reg_range(tok)
        words = [self.is_const(n) for n in range(r[1], r[2] + 1)]
        return None if any(w is None for w in words) else words

    # ---- instruction translation -------------------------------------------------------------
    def translate(self):
        targets = set()
        for off, n, name, asm in self.ins:
            if name.startswith('s_cbranch') or name == 's_branch':
                targets.add(int(parse(asm)[1][0].split('_')[1], 16))
        self.blocks = []          # (offset, [lines], terminator)
        self.uniform_v = set()
        self.has_branches = bool(targets)
        start = 0
        for off, n, name, asm in self.ins:
            if off in targets and off != start:
                self.blocks.append((start, self.lines, None))
                self.lines, start = [], off
            self.pc = off
            term = self.one(name, asm)
            if term:
                self.blocks.append((start, self.lines, term))
                self.lines, start = [], off + n
        if self.lines:
            self.blocks.append((start, self.lines, None))

    def one(self, name, asm):
        _, ops, mods = parse(asm)
        base = re.sub(r'_e(32|64)$', '', name)
        if base in ('s_inst_prefetch', 's_waitcnt', 's_nop', 'v_nop', 's_clause', 's_sopp_0f', 's_wqm_b64', 's_wqm_b32',
                    's_setreg_imm32_b32', 's_code_end', 's_setprio'):
            return None
        if base == 's_endpgm':
            return ('end',)
        if base == 's_branch':
            return ('jump', int(ops[0].split('_')[1], 16))
        if base.startswith('s_cbranch_'):
            cond = {'scc0': '!scc', 'scc1': 'scc', 'vccz': '(vcc & 1) == 0', 'vccnz': '(vcc & 1) != 0',
                    'execz': '(exec & 1) == 0', 'execnz': '(exec & 1) != 0'}[base[10:]]
            return ('branch', cond, int(ops[0].split('_')[1], 16))
        if base == 'exp':
            return self.export(asm)
        handler = getattr(self, 'op_' + base, None)
        if handler:
            return handler(ops, mods)
        if base.startswith('v_cmp'):
            return self.compare(base, ops)
        if base.endswith('_sdwa'):
            return self.sdwa(base[:-len('_sdwa')], ops, mods)
        if base in VEC_F:
            arity, text = VEC_F[base]
            args = [self.f(o) for o in ops[1:1 + arity]]
            if base in ('v_fmac_f32', 'v_mac_f32'):
                args.append(self.f(ops[0]))
            return self.set_vf(reg_range(ops[0])[1], text.format(*args), mods)
        if base in VEC_U:
            arity, text = VEC_U[base]
            if mods:
                raise Unsupported('%s with %s' % (base, mods))
            return self.set_v(reg_range(ops[0])[1], text.format(*[self.u(o) for o in ops[1:1 + arity]]))
        if base in SCA:
            return self.scalar(base, ops)
        raise Unsupported(base)

    # VOP1/VOP2 with sub-dword addressing: select the source fields, run the same operation, then place
    # the low bits of the result in the destination field. The integer forms reuse VEC_U unchanged;
    # float forms are not lowered yet, and dst_unused UNUSED_SEXT has no lowering.
    def sdwa(self, base, ops, mods):
        fields = {}
        for m in mods:
            key, _, value = m.partition(':')
            if key in ('src0_sel', 'src1_sel', 'src2_sel', 'dst_sel', 'dst_unused') and value:
                fields[key] = value
            else:
                raise Unsupported('%s with SDWA modifier %s' % (base, m))
        if base not in VEC_U:
            raise Unsupported('SDWA form of %s' % base)
        arity, text = VEC_U[base]
        sources = []
        for index in range(arity):
            token = ops[1 + index]
            sext = token.startswith('sext(') and token.endswith(')')
            if sext:
                token = token[len('sext('):-1]
            selection = fields.get('src%d_sel' % index)
            if not selection:
                raise Unsupported('%s SDWA without src%d_sel' % (base, index))
            sources.append(sdwa_source(self.u(token), selection, sext))
        result = text.format(*sources)
        select = fields.get('dst_sel')
        if not select:
            raise Unsupported('%s SDWA without dst_sel' % base)
        # The operation's HLSL carries its own operators, so it has to be grouped before the
        # destination field masks it: "v7 | v8 & 0xffff" would otherwise bind as "v7 | (v8 & 0xffff)".
        if SDWA_SEL[select][1] < 32:
            result = '(%s)' % result
        destination = reg_range(ops[0])[1]
        self.set_v(destination, sdwa_dest(result, select, fields.get('dst_unused', 'UNUSED_PAD'),
                                          'v%d' % destination))

    # scalar ALU with descriptor-word bookkeeping
    def scalar(self, base, ops):
        text, scc = SCA[base]
        dst = reg_range(ops[0])
        pair = base.endswith('_b64')
        srcs = ops[1:]
        # 64-bit operands name the low register; exec/vcc name themselves.
        def low(tok):
            r = reg_range(tok)
            return ('s%d' % r[1]) if r and r[0] == 's' and r[1] != r[2] else tok
        srcs = [low(t) for t in srcs]
        dname = low(ops[0])
        if dname == 'm0':
            return None
        syms = [self.sym.get(reg_range(t)[1]) if reg_range(t) and reg_range(t)[0] == 's' else None for t in srcs]
        if any(s and s[0] in ('desc', 'descnum', 'ptr', 'ptrhi') for s in syms):
            if dname in ('exec', 'vcc', 'exec_lo', 'vcc_lo'):
                raise Unsupported('mask from descriptor data')
            d = reg_range(dname)[1]
            own = self.sym.get(d)
            if base in ('s_mov_b32', 's_mov_b64'):
                self.sym[d] = syms[0]
                if pair:
                    hi = self.sym.get(reg_range(ops[1])[2])
                    if hi:
                        self.sym[d + 1] = hi
                    else:
                        self.sym.pop(d + 1, None)
            elif own and own[0] in ('desc', 'ptr', 'ptrhi') and any(s == own for s in syms):
                if own[0] == 'desc':
                    self.pipe.modified.add((own[1], own[2]))
            else:
                self.sym[d] = ('descnum',)
            return None
        values = [self.u(t) for t in srcs]
        expr = text.format(*values)
        if dname in ('exec', 'exec_lo', 'vcc', 'vcc_lo'):
            var = dname[:4].rstrip('_') if dname.startswith('exec') else 'vcc'
            if var == 'exec':
                copy = base in ('s_mov_b64', 's_mov_b32') and self.exec_known and (syms[0] == ('execcopy',))
                if copy:
                    return None
                self.exec_known = False
            self.emit('%s = (%s) & 1;' % (var, expr))
            if scc:
                self.emit('scc = %s;' % scc.replace('$', var))
            return None
        d = reg_range(dname)[1]
        is_mask = pair or any(t in ('exec', 'exec_lo', 'vcc', 'vcc_lo') or (reg_range(t) and reg_range(t)[0] == 's' and reg_range(t)[1] in self.masks)
                              for t in srcs) and base.split('_')[1] in ('mov', 'and', 'or', 'xor', 'andn2', 'orn2', 'nand', 'nor', 'xnor', 'not', 'cselect')
        if base in ('s_mov_b32', 's_mov_b64', 's_movk_i32') and re.fullmatch(r'-?(0x[0-9a-f]+|\d+)', srcs[0]) and srcs[0] not in FLOATS:
            value = int(srcs[0], 0)
            self.set_s(d, values[0], ('const', value & 0xffffffff))
            if pair:
                self.set_s(d + 1, '0x%08xu' % ((value >> 32) & 0xffffffff), ('const', (value >> 32) & 0xffffffff))
            self.masks.discard(d)
            return None
        if base in ('s_mov_b64', 's_mov_b32') and srcs[0] in ('exec', 'exec_lo') and self.exec_known:
            self.set_s(d, 'exec')
            self.sym[d] = ('execcopy',)
            self.masks.add(d)
            if pair:
                self.sym.pop(d + 1, None)
            return None
        if is_mask:
            expr = '(%s) & 1' % expr
            self.masks.add(d)
        else:
            self.masks.discard(d)
        self.set_s(d, expr, scc=scc)
        if pair and base != 's_mov_b64':
            self.set_s(d + 1, '0u')
        elif pair:
            r = reg_range(ops[1])
            self.set_s(d + 1, self.S(r[2]) if r and r[1] != r[2] else '0u')
        return None

    def op_s_and_saveexec_b64(self, ops, mods, text='exec & {0}'):
        d = reg_range(ops[0])[1]
        src = ops[1] if not reg_range(ops[1]) else 's%d' % reg_range(ops[1])[1]
        self.set_s(d, 'exec')
        self.masks.add(d)
        self.exec_known = False
        self.emit('exec = (%s) & 1; scc = exec != 0;' % text.format(self.u(src)))

    op_s_and_saveexec_b32 = op_s_and_saveexec_b64

    def op_s_or_saveexec_b64(self, ops, mods):
        self.op_s_and_saveexec_b64(ops, mods, 'exec | {0}')

    op_s_or_saveexec_b32 = op_s_or_saveexec_b64

    def op_s_andn2_saveexec_b64(self, ops, mods):
        self.op_s_and_saveexec_b64(ops, mods, '{0} & ~exec')

    def op_s_andn2_wrexec_b64(self, ops, mods):
        d = reg_range(ops[0])[1]
        src = 's%d' % reg_range(ops[1])[1]
        self.exec_known = False
        self.emit('exec = (%s & ~exec) & 1; scc = exec != 0;' % self.u(src))
        self.set_s(d, 'exec')
        self.masks.add(d)

    op_s_andn2_wrexec_b32 = op_s_andn2_wrexec_b64

    def op_s_cselect_b32(self, ops, mods):
        self.scalar_select(ops, False)

    def op_s_cselect_b64(self, ops, mods):
        self.scalar_select(ops, True)

    def scalar_select(self, ops, pair):
        low = [('s%d' % reg_range(t)[1]) if reg_range(t) and reg_range(t)[0] == 's' else t for t in ops]
        expr = 'scc ? %s : %s' % (self.u(low[1]), self.u(low[2]))
        if low[0] in ('exec', 'vcc'):
            if low[0] == 'exec':
                self.exec_known = False
            self.emit('%s = (%s) & 1;' % (low[0], expr))
            return
        d = reg_range(low[0])[1]
        self.set_s(d, '(%s) & 1' % expr if pair else expr)
        if pair:
            self.masks.add(d)
            self.set_s(d + 1, '0u')

    def op_s_load_dwordx4(self, ops, mods, count=4):
        dst, base = reg_range(ops[0]), reg_range(ops[1])
        p = self.sym.get(base[1])
        if not p or p[0] != 'ptr':
            raise Unsupported('s_load through s%d holding %s' % (base[1], p))
        try:
            off = int(ops[2], 0)
        except ValueError:
            if p[1] == ('smpheap',) and self.pipe.sampler_heap and count == 4:
                # A sampler picked at run time from the sampler heap: keep the byte offset in the
                # first destination register; image() indexes the heap table with it.
                d = dst[1]
                self.used_s.add(d)
                self.emit('s%d = %s;' % (d, self.u(ops[2])))
                for i in range(count):
                    self.sym[d + i] = ('smpheap', d, i)
                return
            if p[1] == ('smpheap',):
                # Shortcut (default unless X2W_SAMPLER_HEAP=1): a fixed default sampler.
                for i in range(count):
                    self.sym[dst[1] + i] = ('const', 0)
                self.pipe.notes.add('dynamic sampler index replaced by a default sampler')
                return
            raise Unsupported('s_load with a register offset through %s' % (p[1],))
        if p[1][0] == 'root':
            buf = self.pipe.resource((p[1], None), 'buf')
            for i in range(count):
                self.set_s(dst[1] + i, '%s.Load(%d)' % (buf, off + 4 * i))
            return
        for i in range(count):
            self.sym[dst[1] + i] = ('desc', p[1], off, i)

    def op_s_bfm_b32(self, ops, mods):
        """Fold constant bitfield masks: titles build inline sampler words with them."""
        d = reg_range(ops[0])[1]
        vals = []
        for o in ops[1:3]:
            r = reg_range(o)
            if r and r[0] == 's':
                vals.append(self.is_const(r[1]))
            else:
                try:
                    vals.append(int(o, 0))
                except ValueError:
                    vals.append(None)
        if None not in vals:
            value = (((1 << (vals[0] & 31)) - 1) << (vals[1] & 31)) & 0xffffffff
            self.set_s(d, '0x%08xu' % value, ('const', value))
            return None
        self.set_s(d, '(((1u << (%s & 31)) - 1) << (%s & 31))' % (self.u(ops[1]), self.u(ops[2])))
        return None

    def op_s_load_dwordx8(self, ops, mods):
        self.op_s_load_dwordx4(ops, mods, 8)

    def op_s_load_dwordx16(self, ops, mods):
        self.op_s_load_dwordx4(ops, mods, 16)

    def op_s_load_dwordx2(self, ops, mods):
        self.op_s_load_dwordx4(ops, mods, 2)

    def op_s_load_dword(self, ops, mods):
        self.op_s_load_dwordx4(ops, mods, 1)

    def op_s_buffer_load_dword(self, ops, mods, count=1):
        dst = reg_range(ops[0])
        buf = self.pipe.resource(self.descriptor(ops[1]), 'buf')
        offset = self.u(ops[2])
        for i in range(count):
            self.set_s(dst[1] + i, '%s.Load(%s + %d)' % (buf, offset, 4 * i))
            if not mods and re.fullmatch(r'0x[0-9a-f]+|\d+', ops[2]):
                self.sym[dst[1] + i] = ('fixed_buffer_word',)

    def op_s_buffer_load_dwordx2(self, ops, mods):
        self.op_s_buffer_load_dword(ops, mods, 2)

    def op_s_buffer_load_dwordx4(self, ops, mods):
        self.op_s_buffer_load_dword(ops, mods, 4)

    def op_s_buffer_load_dwordx8(self, ops, mods):
        self.op_s_buffer_load_dword(ops, mods, 8)

    def op_s_buffer_load_dwordx16(self, ops, mods):
        self.op_s_buffer_load_dword(ops, mods, 16)

    def buffer_address(self, ops, mods):
        key = self.descriptor(ops[2])
        buf = self.pipe.resource(key, 'buf')
        offset = int(ops[3], 0)
        for m in mods:
            if m.startswith('offset:'):
                offset += int(m[7:], 0)
        if 'idxen' in mods:
            if key in self.pipe.modified:
                raise Unsupported('indexed load through a modified descriptor')
            address = '%s * ud%d + %d' % (self.u(ops[1]), self.pipe.ud(('stride',) + (self.pipe.resources[key]['index'],)), offset)
        elif 'offen' in mods:
            address = '%s + %d' % (self.u(ops[1]), offset)
        else:
            address = '%d' % offset
        return buf, address

    def op_buffer_load_dword(self, ops, mods, count=1):
        dst = reg_range(ops[0])
        buf, address = self.buffer_address(ops, mods)
        self.emit('a = %s;' % address)
        for i in range(count):
            self.set_v(dst[1] + i, '%s.Load(a + %d)' % (buf, 4 * i))

    def op_buffer_load_dwordx2(self, ops, mods):
        self.op_buffer_load_dword(ops, mods, 2)

    def op_buffer_load_dwordx3(self, ops, mods):
        self.op_buffer_load_dword(ops, mods, 3)

    def op_buffer_load_dwordx4(self, ops, mods):
        self.op_buffer_load_dword(ops, mods, 4)

    def tbuffer(self, ops, mods, count):
        dst = reg_range(ops[0])
        formats = [m for m in mods if m.startswith('format:')]
        fmt = formats[0][8:-1] if formats else 'BUF_FMT_8_UNORM'  # the disassembler omits the default format
        buf, address = self.buffer_address(ops, [m for m in mods if not m.startswith('format:')])
        self.emit('a = %s;' % address)
        table = {'BUF_FMT_8_8_8_8_UNORM': ('ldu({0}, a)', ['U((w & 255) / 255.0)', 'U(((w >> 8) & 255) / 255.0)', 'U(((w >> 16) & 255) / 255.0)', 'U((w >> 24) / 255.0)']),
                 'BUF_FMT_8_8_UNORM': ('ldu({0}, a)', ['U((w & 255) / 255.0)', 'U(((w >> 8) & 255) / 255.0)', '0u', 'U(1.0)']),
                 'BUF_FMT_8_UNORM': ('ldu({0}, a)', ['U((w & 255) / 255.0)', '0u', '0u', 'U(1.0)']),
                 'BUF_FMT_8_UINT': ('{0}.Load(a & ~3u) >> ((a & 3u) * 8u)', ['w & 255u', '0u', '0u', '1u']),
                 'BUF_FMT_16_16_UNORM': ('ldu({0}, a)', ['U((w & 65535) / 65535.0)', 'U((w >> 16) / 65535.0)', '0u', 'U(1.0)']),
                 'BUF_FMT_16_16_FLOAT': ('ldu({0}, a)', ['U(f16tof32(w))', 'U(f16tof32(w >> 16))', '0u', 'U(1.0)']),
                 'BUF_FMT_32_32_32_32_FLOAT': (None, ['{0}.Load(a)', '{0}.Load(a + 4)', '{0}.Load(a + 8)', '{0}.Load(a + 12)']),
                 'BUF_FMT_32_32_32_FLOAT': (None, ['{0}.Load(a)', '{0}.Load(a + 4)', '{0}.Load(a + 8)', 'U(1.0)']),
                 'BUF_FMT_32_32_FLOAT': (None, ['{0}.Load(a)', '{0}.Load(a + 4)', '0u', 'U(1.0)']),
                 'BUF_FMT_32_FLOAT': (None, ['{0}.Load(a)', '0u', '0u', 'U(1.0)'])}
        if fmt not in table:
            raise Unsupported('buffer format %s' % fmt)
        load, comps = table[fmt]
        if load:
            self.emit('w = %s;' % load.format(buf))
        for i in range(count):
            self.set_v(dst[1] + i, comps[i].format(buf))

    def op_tbuffer_load_format_x(self, ops, mods):
        self.tbuffer(ops, mods, 1)

    def op_tbuffer_load_format_xy(self, ops, mods):
        self.tbuffer(ops, mods, 2)

    def op_tbuffer_load_format_xyz(self, ops, mods):
        self.tbuffer(ops, mods, 3)

    def op_tbuffer_load_format_xyzw(self, ops, mods):
        self.tbuffer(ops, mods, 4)

    def image(self, ops, mods, how):
        dst = reg_range(ops[0])
        dim = [m for m in mods if m.startswith('dim:')][0][4:]
        dmask = int([m for m in mods if m.startswith('dmask:')][0][6:], 0)
        if dim not in ('SQ_RSRC_IMG_2D', 'SQ_RSRC_IMG_3D'):
            raise Unsupported('image dimension %s' % dim)
        three = dim == 'SQ_RSRC_IMG_3D'
        coords = ops[1]
        if coords.startswith('['):
            cs = [c.strip() for c in coords[1:-1].split(',')]
        else:
            r = reg_range(coords)
            cs = ['v%d' % n for n in range(r[1], r[2] + 1)]
        tex = self.pipe.resource(self.descriptor(ops[2]), 'tex3d' if three else 'tex2d')
        if how == 'load':
            if three:
                call = '%s.Load(int4(%s, %s, %s, 0))' % (tex, self.u(cs[0]), self.u(cs[1]), self.u(cs[2]))
            else:
                call = '%s.Load(int3(%s, %s, 0))' % (tex, self.u(cs[0]), self.u(cs[1]))
        else:
            heap = self.heap_sampler(ops[3])
            words = None if heap else self.const_words(ops[3])
            smp = heap or self.pipe.sampler(tuple(words) if words else self.descriptor(ops[3]))
            uv = ('float3(%s, %s, %s)' % (self.f(cs[0]), self.f(cs[1]), self.f(cs[2])) if three
                  else 'float2(%s, %s)' % (self.f(cs[0]), self.f(cs[1])))
            if how == 'lz' or not self.exec_known or self.kind != 'ps' or len(self.blocks) > 0:
                call = '%s.SampleLevel(%s, %s, 0)' % (tex, smp, uv)
            else:
                call = '%s.Sample(%s, %s)' % (tex, smp, uv)
        self.emit('t = %s;' % call)
        n = dst[1]
        for bit, comp in enumerate('xyzw'):
            if dmask & (1 << bit):
                self.set_v(n, 'U(t.%s)' % comp)
                n += 1

    def op_image_sample(self, ops, mods):
        self.image(ops, mods, 'sample')

    def op_image_sample_lz(self, ops, mods):
        self.image(ops, mods, 'lz')

    def op_image_load(self, ops, mods):
        self.image(ops, mods, 'load')

    def op_v_cvt_i32_f32(self, ops, mods):
        if mods:
            raise Unsupported('v_cvt_i32_f32 with modifiers')
        dst = reg_range(ops[0])[1]
        self.set_v(dst, 'uint(int(F(%s)))' % self.u(ops[1]))
        src = reg_range(ops[1])
        if self.exec_known and src and src[0] == 's' and self.sym.get(src[1]) == ('fixed_buffer_word',):
            self.uniform_v.add(dst)

    def op_v_readfirstlane_b32(self, ops, mods):
        src, dst = reg_range(ops[1]), reg_range(ops[0])
        if (mods or self.has_branches or not self.exec_known or not dst or dst[0] != 's' or
                not src or src[0] != 'v' or src[1] not in self.uniform_v):
            raise Unsupported('v_readfirstlane_b32 requires a proven uniform source')
        self.set_s(reg_range(ops[0])[1], self.u(ops[1]))

    def op_v_mov_b32(self, ops, mods):
        self.set_v(reg_range(ops[0])[1], self.u(ops[1]))

    def op_v_cndmask_b32(self, ops, mods):
        mask = self.u(ops[3]) if len(ops) > 3 else 'vcc'
        self.set_v(reg_range(ops[0])[1], '(%s & 1) ? %s : %s' % (mask, self.u(ops[2]), self.u(ops[1])))

    def op_v_interp_p1_f32(self, ops, mods):
        self.note_attr(ops[2], 'linear')

    def op_v_interp_p2_f32(self, ops, mods):
        attr, comp = self.note_attr(ops[2], 'linear')
        self.set_v(reg_range(ops[0])[1], 'U(i.param%d.%s)' % (attr, comp))

    def op_v_interp_mov_f32(self, ops, mods):
        attr, comp = self.note_attr(ops[2], 'flat')
        self.set_v(reg_range(ops[0])[1], 'U(i.fparam%d.%s)' % (attr, comp))

    def note_attr(self, tok, how):
        m = re.fullmatch(r'attr(\d+)\.([xyzw])', tok)
        attr = int(m.group(1))
        self.attrs.setdefault(attr, set()).add(how)
        return attr, m.group(2)

    def mix(self, ops, mods, high):
        """v_fma_mix{lo,hi}_f16: fma of operands that are f32 or f16 halves; the f16 result goes to one half."""
        sel = [0, 0, 0]
        for m in mods:
            if m.startswith('op_sel_hi:'):
                sel = [int(x) for x in m[11:-1].split(',')]
            elif m.startswith('op_sel:'):
                raise Unsupported('op_sel on fma_mix')
        def operand(tok, is_half):
            if not is_half or tok in ('-0', '0'):
                return self.f(tok)
            return 'f16tof32(%s)' % self.u(tok)
        expr = '%s * %s + %s' % tuple(operand(t, s) for t, s in zip(ops[1:4], sel))
        if 'clamp' in mods:
            expr = 'saturate(%s)' % expr
        d = reg_range(ops[0])[1]
        self.used_v.add(d)
        if high:
            self.set_v(d, '(v%d & 0xffffu) | (f32tof16(%s) << 16)' % (d, expr))
        else:
            self.set_v(d, '(v%d & 0xffff0000u) | f32tof16(%s)' % (d, expr))

    def op_v_fma_mixlo_f16(self, ops, mods):
        self.mix(ops, mods, False)

    def op_v_fma_mixhi_f16(self, ops, mods):
        self.mix(ops, mods, True)

    def op_v_cvt_pkrtz_f16_f32(self, ops, mods):
        self.set_v(reg_range(ops[0])[1], 'f32tof16(%s) | (f32tof16(%s) << 16)' % (self.f(ops[1]), self.f(ops[2])))

    def compare(self, base, ops):
        m = re.fullmatch(r'v_cmp(x?)_(\w+?)_(f32|u32|i32)(_sdwa)?', base)
        if not m or m.group(4):
            raise Unsupported(base)
        writes_exec, cond, typ = m.group(1) == 'x', m.group(2), m.group(3)
        if len(ops) == 3:
            dst, a, b = ops
        else:
            dst, (a, b) = ('exec' if writes_exec else 'vcc'), ops
        if typ == 'f32':
            x, y = self.f(a), self.f(b)
        elif typ == 'i32':
            x, y = 'int(%s)' % self.u(a), 'int(%s)' % self.u(b)
        else:
            x, y = self.u(a), self.u(b)
        table = {'lt': '{0} < {1}', 'eq': '{0} == {1}', 'le': '{0} <= {1}', 'gt': '{0} > {1}', 'lg': '{0} != {1}', 'ne': '{0} != {1}',
                 'ge': '{0} >= {1}', 'neq': '!({0} == {1})', 'nlt': '!({0} < {1})', 'nle': '!({0} <= {1})', 'ngt': '!({0} > {1})',
                 'nge': '!({0} >= {1})', 'nlg': '!({0} != {1})', 'o': '({0} == {0} && {1} == {1})', 'u': '({0} != {0} || {1} != {1})',
                 't': 'true', 'f': 'false'}
        if cond not in table:
            raise Unsupported(base)
        expr = '((%s) ? 1u : 0u)' % table[cond].format(x, y)
        if not self.exec_known:
            expr = '(exec & %s)' % expr
        if writes_exec or dst in ('exec', 'exec_lo'):
            self.exec_known = False
            self.emit('exec = %s;' % expr)
        elif dst in ('vcc', 'vcc_lo'):
            self.emit('vcc = %s;' % expr)
        else:
            d = reg_range(dst)[1]
            self.set_s(d, expr)
            self.masks.add(d)

    def export(self, asm):
        m = re.fullmatch(r'exp (\w+) (.*?)((?: done| compr| vm)*)', asm)
        target, vals, flags = m.group(1), [v.strip() for v in m.group(2).split(',')], m.group(3).split()
        if 'vm' in flags and not self.exec_known:
            self.emit('if (!(exec & 1)) discard;')
        if 'compr' in flags:
            a, b = self.u(vals[0]), self.u(vals[2])
            exprs = ['f16tof32(%s)' % a, 'f16tof32(%s >> 16)' % a, 'f16tof32(%s)' % b, 'f16tof32(%s >> 16)' % b]
        else:
            exprs = ['0.0' if v == 'off' else 'F(%s)' % self.u(v) for v in vals]
        name = {'pos0': 'pos'}.get(target, target)
        if not re.fullmatch(r'pos|param\d+|mrt[0-7]', name):
            raise Unsupported('export target %s' % target)
        self.exports[name] = True
        self.emit('o.%s = float4(%s);' % (name, ', '.join(exprs)))
        return None

    # ---- output ------------------------------------------------------------------------------
    def body(self):
        """Statements of the whole program as structured HLSL."""
        out = []
        offsets = [b[0] for b in self.blocks]
        index = {off: i for i, off in enumerate(offsets)}
        loops = any(t and t[0] in ('jump', 'branch') and t[-1] <= b[0] for b in self.blocks for t in [b[2]])
        if len(self.blocks) == 1:
            return ['    ' + line for line in self.blocks[0][1]]

        def goto(off):
            if off not in index:
                raise Unsupported('branch into the middle of a block (+0x%x)' % off)
            return index[off]
        out.append('    uint pc = 0;')
        if loops:
            out.append('    [loop] for (uint guard = 0; guard < 65536 && pc != 0xffffu; ++guard) {')
        for i, (off, lines, term) in enumerate(self.blocks):
            out.append('    %sif (pc == %d) {' % ('else ' if loops and i else '', i))
            out += ['        ' + line for line in lines]
            nxt = i + 1 if i + 1 < len(self.blocks) else 0xffff
            if not term:
                out.append('        pc = %d;' % nxt)
            elif term[0] == 'end':
                out.append('        pc = 0xffffu;')
            elif term[0] == 'jump':
                out.append('        pc = %d;' % goto(term[1]))
            else:
                out.append('        pc = (%s) ? %d : %d;' % (term[1], goto(term[2]), nxt))
            out.append('    }')
        if loops:
            out.append('    }')
        return out

    def declarations(self):
        lines = []
        if self.used_s:
            lines.append('    uint %s;' % ', '.join('s%d = 0' % n for n in sorted(self.used_s)))
        if self.used_v:
            lines.append('    uint %s;' % ', '.join('v%d = 0' % n for n in sorted(self.used_v)))
        lines.append('    uint vcc = 0, exec = 1, m0 = 0, a = 0, w = 0; bool scc = false; float4 t = 0;')
        return lines


# name -> (source operand count, expression); operands are floats
VEC_F = {
    'v_add_f32': (2, '{0} + {1}'), 'v_sub_f32': (2, '{0} - {1}'), 'v_subrev_f32': (2, '{1} - {0}'),
    'v_mul_f32': (2, '{0} * {1}'), 'v_mul_legacy_f32': (2, 'mul_legacy({0}, {1})'),
    'v_min_f32': (2, 'min({0}, {1})'), 'v_max_f32': (2, 'max({0}, {1})'),
    'v_fmac_f32': (2, '{0} * {1} + {2}'), 'v_mac_f32': (2, '{0} * {1} + {2}'),
    'v_fma_f32': (3, '{0} * {1} + {2}'), 'v_mad_f32': (3, '{0} * {1} + {2}'), 'v_mad_legacy_f32': (3, 'mul_legacy({0}, {1}) + {2}'),
    'v_fmamk_f32': (3, '{0} * {1} + {2}'), 'v_fmaak_f32': (3, '{0} * {1} + {2}'),
    'v_madmk_f32': (3, '{0} * {1} + {2}'), 'v_madak_f32': (3, '{0} * {1} + {2}'),
    'v_rcp_f32': (1, '1.0 / {0}'), 'v_rcp_iflag_f32': (1, '1.0 / {0}'), 'v_rsq_f32': (1, 'rsqrt({0})'), 'v_sqrt_f32': (1, 'sqrt({0})'),
    'v_exp_f32': (1, 'exp2({0})'), 'v_log_f32': (1, 'log2({0})'), 'v_floor_f32': (1, 'floor({0})'), 'v_ceil_f32': (1, 'ceil({0})'),
    'v_trunc_f32': (1, 'trunc({0})'), 'v_fract_f32': (1, 'frac({0})'), 'v_rndne_f32': (1, 'round({0})'),
    'v_sin_f32': (1, 'sin({0} * 6.283185307)'), 'v_cos_f32': (1, 'cos({0} * 6.283185307)'),
    'v_med3_f32': (3, 'med3({0}, {1}, {2})'), 'v_min3_f32': (3, 'min({0}, min({1}, {2}))'), 'v_max3_f32': (3, 'max({0}, max({1}, {2}))'),
}
# operands are uints; the result is a uint expression
VEC_U = {
    'v_add_nc_u32': (2, '{0} + {1}'), 'v_sub_nc_u32': (2, '{0} - {1}'), 'v_subrev_nc_u32': (2, '{1} - {0}'),
    'v_mul_lo_u32': (2, '{0} * {1}'), 'v_and_b32': (2, '{0} & {1}'), 'v_or_b32': (2, '{0} | {1}'), 'v_xor_b32': (2, '{0} ^ {1}'),
    'v_not_b32': (1, '~{0}'), 'v_lshlrev_b32': (2, '{1} << ({0} & 31)'), 'v_lshrrev_b32': (2, '{1} >> ({0} & 31)'),
    'v_ashrrev_i32': (2, 'uint(int({1}) >> ({0} & 31))'), 'v_min_u32': (2, 'min({0}, {1})'), 'v_max_u32': (2, 'max({0}, {1})'),
    'v_min_i32': (2, 'uint(min(int({0}), int({1})))'), 'v_max_i32': (2, 'uint(max(int({0}), int({1})))'),
    'v_cvt_f32_u32': (1, 'U(float({0}))'), 'v_cvt_f32_i32': (1, 'U(float(int({0})))'),
    'v_cvt_u32_f32': (1, 'uint(max(F({0}), 0.0))'),
    'v_cvt_f32_ubyte0': (1, 'U(float({0} & 255))'), 'v_cvt_f32_ubyte1': (1, 'U(float(({0} >> 8) & 255))'),
    'v_cvt_f32_ubyte2': (1, 'U(float(({0} >> 16) & 255))'), 'v_cvt_f32_ubyte3': (1, 'U(float({0} >> 24))'),
    'v_cvt_f32_f16': (1, 'U(f16tof32({0}))'), 'v_cvt_f16_f32': (1, 'f32tof16(F({0}))'),
    'v_bfe_u32': (3, '({0} >> ({1} & 31)) & ((1u << ({2} & 31)) - 1)'), 'v_mad_u32_u24': (3, '({0} & 0xffffffu) * ({1} & 0xffffffu) + {2}'),
    'v_lshl_add_u32': (3, '({0} << ({1} & 31)) + {2}'), 'v_add_lshl_u32': (3, '({0} + {1}) << ({2} & 31)'), 'v_add3_u32': (3, '{0} + {1} + {2}'),
    'v_or3_b32': (3, '{0} | {1} | {2}'), 'v_and_or_b32': (3, '({0} & {1}) | {2}'), 'v_lshl_or_b32': (3, '({0} << ({1} & 31)) | {2}'),
    'v_xad_u32': (3, '({0} ^ {1}) + {2}'), 'v_bfi_b32': (3, '({0} & {1}) | (~{0} & {2})'),
    'v_mul_u32_u24': (2, '({0} & 0xffffffu) * ({1} & 0xffffffu)'),
}
# scalar ops: (expression over uint sources, SCC rule with $ for the result or None)
SCA = {
    's_mov_b32': ('{0}', None), 's_mov_b64': ('{0}', None), 's_movk_i32': ('{0}', None),
    's_and_b32': ('{0} & {1}', '$ != 0'), 's_or_b32': ('{0} | {1}', '$ != 0'), 's_xor_b32': ('{0} ^ {1}', '$ != 0'),
    's_andn2_b32': ('{0} & ~{1}', '$ != 0'), 's_orn2_b32': ('{0} | ~{1}', '$ != 0'), 's_not_b32': ('~{0}', '$ != 0'),
    's_and_b64': ('{0} & {1}', '$ != 0'), 's_or_b64': ('{0} | {1}', '$ != 0'), 's_xor_b64': ('{0} ^ {1}', '$ != 0'),
    's_andn2_b64': ('{0} & ~{1}', '$ != 0'), 's_orn2_b64': ('{0} | ~{1}', '$ != 0'), 's_not_b64': ('~{0}', '$ != 0'),
    's_nand_b64': ('~({0} & {1})', '$ != 0'), 's_nor_b64': ('~({0} | {1})', '$ != 0'), 's_xnor_b64': ('~({0} ^ {1})', '$ != 0'),
    's_lshl_b32': ('{0} << ({1} & 31)', '$ != 0'), 's_lshr_b32': ('{0} >> ({1} & 31)', '$ != 0'),
    's_ashr_i32': ('uint(int({0}) >> ({1} & 31))', '$ != 0'),
    's_add_i32': ('{0} + {1}', None), 's_add_u32': ('{0} + {1}', None), 's_sub_i32': ('{0} - {1}', None), 's_sub_u32': ('{0} - {1}', None),
    's_mul_i32': ('{0} * {1}', None), 's_min_u32': ('min({0}, {1})', None), 's_max_u32': ('max({0}, {1})', None),
    's_min_i32': ('uint(min(int({0}), int({1})))', None), 's_max_i32': ('uint(max(int({0}), int({1})))', None),
    's_bfe_u32': ('({0} >> ({1} & 31)) & ((1u << (({1} >> 16) & 127)) - 1)', '$ != 0'),
    's_brev_b32': ('reversebits({0})', None),
    # s_bfm_b32: a mask of S0[4:0] ones shifted left by S1[4:0]; SCC unchanged.
    's_bfm_b32': ('(((1u << ({0} & 31)) - 1) << ({1} & 31))', None),
}
for _name, _op in (('eq', '=='), ('lg', '!='), ('gt', '>'), ('ge', '>='), ('lt', '<'), ('le', '<=')):
    for _typ, _cast in (('i32', 'int({0}) %s int({1})'), ('u32', '{0} %s {1}')):
        def _make(op=_op, cast=_cast):
            def handler(self, ops, mods):
                self.emit('scc = %s;' % (cast % op).format(self.u(ops[0]), self.u(ops[1])))
            return handler
        setattr(Stage, 'op_s_cmp_%s_%s' % (_name, _typ), _make())
        setattr(Stage, 'op_s_cmpk_%s_%s' % (_name, _typ), _make())


def parse_root(blob):
    if not blob or blob[:4] != b'DXBC':
        raise Unsupported('root signature blob missing')
    n, = struct.unpack_from('<I', blob, 28)
    for o in struct.unpack_from('<%dI' % n, blob, 32):
        if blob[o:o + 4] == b'RTS0':
            d = blob[o + 8:]
            ver, nparam, poff, nsamp, soff, flags = struct.unpack_from('<6I', d, 0)
            params = []
            for i in range(nparam):
                ptype, vis, off = struct.unpack_from('<3I', d, poff + 12 * i)
                p = dict(type=ptype, vis=vis)
                if ptype == 1:
                    p['reg'], p['space'], p['count'] = struct.unpack_from('<3I', d, off)
                params.append(p)
            return dict(flags=flags, params=params)
    raise Unsupported('no RTS0 part')


# Run-time sampler heap indexing: an s_load from the sampler heap with a register offset becomes a
# read of a bounded table, SamplerState qh[SAMPLER_HEAP_SIZE] : register(s0, space1), the form a
# desktop SM 5.1 / 6.0 shader can index at run time. Opt-in with X2W_SAMPLER_HEAP=1 until the
# stand-in binds that table; otherwise such a load keeps the default-sampler shortcut.
#
# Contract at the manifest boundary ('smpheap <count> <space> fallback <entry>'):
# - entries 0 .. SAMPLER_HEAP_FALLBACK - 1 hold the title's sampler-heap entries, entry n at byte
#   offset 16 * n;
# - entry SAMPLER_HEAP_FALLBACK is reserved: the host fills it with a fallback sampler;
# - the shader computes the entry itself (x2w_sampler_entry): a byte offset that is not a multiple
#   of 16, or that lies at or past 16 * SAMPLER_HEAP_FALLBACK, reads the fallback entry. The host
#   cannot check the offset, which is computed in the shader, so the guard is in the shader.
SAMPLER_HEAP_SIZE = 2048  # D3D12's limit for a shader-visible sampler heap
SAMPLER_HEAP_SPACE = 1
SAMPLER_HEAP_FALLBACK = SAMPLER_HEAP_SIZE - 1


class Pipeline:
    def __init__(self, vs_code, ps_code, root_blob, sampler_heap=None):
        self.root = parse_root(root_blob)
        self.sampler_heap = os.environ.get('X2W_SAMPLER_HEAP') == '1' if sampler_heap is None else sampler_heap
        self.uses_sampler_heap = False
        self.resources = {}   # key -> dict(index, kind)
        self.samplers = {}    # key -> index
        self.uds = []         # entry-constant sources
        self.notes = set()    # shortcuts taken (written to the manifest as comments)
        self.modified = set()
        self.rect = self.ud(('rect',))
        self.vs = Stage(self, 'vs', vs_code)
        self.ps = Stage(self, 'ps', ps_code)
        self.params = sorted(int(k[5:]) for k in self.vs.exports if k.startswith('param'))
        self.flat = sorted(a for a, how in self.ps.attrs.items() if 'flat' in how)
        missing = [a for a in self.ps.attrs if a not in self.params]
        if missing:
            raise Unsupported('pixel stage reads attributes %s the vertex stage does not export' % missing)
        if 'pos' not in self.vs.exports:
            raise Unsupported('vertex stage exports no position')

    def ud(self, source):
        if source not in self.uds:
            self.uds.append(source)
        return self.uds.index(source)

    def resource(self, key, kind):
        r = self.resources.setdefault(key, dict(index=len(self.resources), kind=kind))
        if r['kind'] != kind:
            raise Unsupported('resource %s used as %s and %s' % (key, r['kind'], kind))
        return 'r%d' % r['index']

    def sampler(self, key):
        if not isinstance(key[0], int):
            raise Unsupported('sampler from a descriptor')
        return 'q%d' % self.samplers.setdefault(key, len(self.samplers))

    def common(self):
        out = ['#define F(x) asfloat(x)', '#define U(x) asuint(x)',
               'float mul_legacy(float a, float b) { return (a == 0.0 || b == 0.0) ? 0.0 : a * b; }',
               'float med3(float a, float b, float c) { return max(min(a, b), min(max(a, b), c)); }',
               'uint ldu(ByteAddressBuffer b, uint a) { uint k = (a & 3) * 8, lo = b.Load(a & ~3u); if (k == 0) return lo; '
               'return (lo >> k) | (b.Load((a & ~3u) + 4) << (32 - k)); }',
               'cbuffer entry : register(b0) { %s };' % ' '.join('uint ud%d;' % i for i in range(len(self.uds)))]
        for key, r in sorted(self.resources.items(), key=lambda kv: kv[1]['index']):
            out.append('%s r%d : register(t%d);' % ({'buf': 'ByteAddressBuffer', 'tex2d': 'Texture2D<float4>', 'tex3d': 'Texture3D<float4>'}[r['kind']], r['index'], r['index']))
        for key, i in self.samplers.items():
            out.append('SamplerState q%d : register(s%d);' % (i, i))
        if self.uses_sampler_heap:
            out.append('SamplerState qh[%d] : register(s0, space%d);' % (SAMPLER_HEAP_SIZE, SAMPLER_HEAP_SPACE))
            out.append('uint x2w_sampler_entry(uint offset) { return (offset & 15u) != 0 || offset >= %du ? %du : offset >> 4; }'
                       % (16 * SAMPLER_HEAP_FALLBACK, SAMPLER_HEAP_FALLBACK))
        return out

    def interface(self, for_pixel):
        fields = []
        for p in range(max(self.params) + 1 if self.params else 0):
            fields.append('    float4 param%d : PARAM%d;' % (p, p))
        # Attributes read without interpolation get a second, flat copy of the vertex output.
        for p in self.flat:
            fields.append('    nointerpolation float4 fparam%d : FPARAM%d;' % (p, p))
        fields.append('    float4 pos : SV_Position;')
        return fields

    def hlsl_vs(self):
        s = self.vs
        out = self.common() + ['struct VSOut {'] + self.interface(False) + ['};', 'VSOut body(uint vid) {', '    VSOut o = (VSOut)0;']
        s.used_v.add(0)
        out += s.declarations() + ['    v0 = vid;'] + s.body() + ['    o.fparam%d = o.param%d;' % (p, p) for p in self.flat] + ['    return o;', '}',
            '// Rectangle lists arrive as six vertices per rectangle: corners 0 1 2, 0 2 and the fourth corner 0 + 2 - 1.',
            'VSOut main(uint vid : SV_VertexID) {',
            '    if (ud%d == 0) return body(vid);' % self.rect,
            '    uint first = (vid / 6) * 3, corner = vid % 6;',
            '    if (corner < 3) return body(first + corner);',
            '    if (corner == 3) return body(first);',
            '    if (corner == 4) return body(first + 2);',
            '    VSOut p = body(first), q = body(first + 1), r = body(first + 2), o;']
        for p in range(max(self.params) + 1 if self.params else 0):
            out.append('    o.param%d = p.param%d + r.param%d - q.param%d;' % (p, p, p, p))
        out += ['    o.fparam%d = p.fparam%d;' % (p, p) for p in self.flat]
        out += ['    o.pos = p.pos + r.pos - q.pos;', '    return o;', '}']
        return '\n'.join(out) + '\n'

    def hlsl_ps(self):
        s = self.ps
        targets = sorted(int(k[3:]) for k in s.exports if k.startswith('mrt'))
        out = self.common() + ['struct PSIn {'] + self.interface(True) + ['};', 'struct PSOut {']
        out += ['    float4 mrt%d : SV_Target%d;' % (t, t) for t in targets] + ['};', 'PSOut main(PSIn i) {', '    PSOut o = (PSOut)0;']
        out += s.declarations() + s.body() + ['    return o;', '}']
        return '\n'.join(out) + '\n'

    def manifest(self):
        lines = ['x2w-manifest 1', 'ud %d' % len(self.uds)]
        for i, source in enumerate(self.uds):
            lines.append('udsrc %d %s' % (i, ' '.join(str(x) for x in source)))
        for key, r in sorted(self.resources.items(), key=lambda kv: kv[1]['index']):
            source, off = key
            where = {'table': 'table %d %d' % (source[1] if len(source) > 1 else 0, off or 0), 'vbtab': 'vb %d' % ((off or 0) // 16),
                     'root': 'root %d' % (source[1] if len(source) > 1 else 0)}.get(source[0])
            if where is None:
                raise Unsupported('resource through %s' % (source,))
            lines.append('srv %d %s %s' % (r['index'], r['kind'], where))
        for key, i in self.samplers.items():
            lines.append('smp %d %s' % (i, ' '.join('0x%x' % w for w in key)))
        if self.uses_sampler_heap:
            lines.append('smpheap %d %d fallback %d' % (SAMPLER_HEAP_SIZE, SAMPLER_HEAP_SPACE, SAMPLER_HEAP_FALLBACK))
        lines.append('params %d' % (max(self.params) + 1 if self.params else 0))
        lines.append('targets %s' % ' '.join(k[3:] for k in sorted(self.ps.exports) if k.startswith('mrt')))
        return '\n'.join(lines) + '\n'


def fnv(data, h=1469598103934665603):
    for b in data:
        h = ((h ^ b) * 1099511628211) & 0xffffffffffffffff
    return h


def pipeline_key(vs_code, ps_code, root_blob):
    """Must match the runtime: FNV-1a over the three blobs, chained."""
    return fnv(root_blob, fnv(ps_code, fnv(vs_code)))


FXC = os.environ.get('X2W_FXC', r'C:\Program Files (x86)\Windows Kits\10\bin\10.0.26100.0\x64\fxc.exe')


def translate_to(out_dir, key, vs_code, ps_code, root_blob, check=True):
    """Writes <key>.vs.hlsl, <key>.ps.hlsl and <key>.txt (last), or <key>.fail with the reason."""
    name = os.path.join(out_dir, '%016x' % key)
    try:
        pipe = Pipeline(vs_code, ps_code, root_blob)
        vs, ps, manifest = pipe.hlsl_vs(), pipe.hlsl_ps(), pipe.manifest()
    except Unsupported as reason:
        open(name + '.fail', 'w').write(str(reason) + '\n')
        return 'unsupported: %s' % reason
    open(name + '.vs.hlsl', 'w').write(vs)
    open(name + '.ps.hlsl', 'w').write(ps)
    if check and os.path.exists(FXC):
        import subprocess
        for stage, profile in (('vs', 'vs_5_1'), ('ps', 'ps_5_1')):
            r = subprocess.run([FXC, '/nologo', '/T', profile, '/E', 'main', '/Fo', name + '.%s.cso' % stage, name + '.%s.hlsl' % stage],
                               capture_output=True, text=True)
            if r.returncode:
                open(name + '.fail', 'w').write('fxc %s: %s\n' % (stage, r.stderr[-2000:]))
                return 'fxc %s failed: %s' % (stage, r.stderr.strip().splitlines()[-1] if r.stderr.strip() else '?')
    open(name + '.txt', 'w').write(manifest)
    return 'ok'


def classify(blob):
    text = ' '.join(x[3] for x in decode(strip_header(blob)) if x[2] == 'exp')
    return 'vs' if 'pos' in text else 'ps' if 'mrt' in text else 'other'


def translate_request(out_dir, key):
    """Translate <dir>/req_<key>.{a,b,root} (the two code blobs of a pipeline, in description order)."""
    name = os.path.join(out_dir, 'req_%016x' % key)
    a, b, root = (open(name + ext, 'rb').read() for ext in ('.a', '.b', '.root'))
    try:
        if a.startswith(b'DXBC') or b.startswith(b'DXBC'):
            # Slot 86 records complete containers in canonical (VS, PS) order.
            # Keep the container bytes in the cache key; translate only the bounded native code.
            import zlib
            from pixel_aot import container_code
            from resource_inventory import parts
            try:
                a, b = (container_code(parts(blob)[b'DXIL'], stage=stage)
                        for blob, stage in ((a, 1), (b, 0)))
            except (KeyError, ValueError, struct.error, zlib.error) as error:
                raise Unsupported('invalid stream shader: %s' % error) from error
            if classify(a) != 'vs' or classify(b) != 'ps':
                raise Unsupported('stream stage and shader export disagree')
            kinds = {'vs': a, 'ps': b}
        else:
            kinds = {classify(a): a, classify(b): b}
    except Unsupported as reason:
        open(os.path.join(out_dir, '%016x.fail' % key), 'w').write(str(reason) + '\n')
        return 'unsupported: %s' % reason
    if set(kinds) != {'vs', 'ps'}:
        open(os.path.join(out_dir, '%016x.fail' % key), 'w').write('stages %s\n' % sorted(kinds))
        return 'stages %s' % sorted(kinds)
    return translate_to(out_dir, key, kinds['vs'], kinds['ps'], root)


def main():
    out_dir = sys.argv[2]
    os.makedirs(out_dir, exist_ok=True)
    if sys.argv[1] == '--capture':
        # --capture <dir> <capture.bin>: write request files for every pipeline the capture binds
        import capture
        cap = capture.Capture(sys.argv[3])
        done = set()
        for c in cap.calls:
            w = c['args'][0]
            if c['slot'] != 25 or w in done:
                continue
            done.add(w)
            p = cap.psos.get(w)
            while p and p['path'] == 55 and p['base']:
                p = cap.psos.get(p['base'])
            if not p or p['path'] != 57:
                continue
            q = struct.unpack_from('<12Q', p['desc'], 0)
            a, b = cap.memory.read(q[5], q[6]), cap.memory.read(q[9], q[10])
            root = cap.roots.get(q[0], b'')
            if not a or not b or len(a) != q[6] or len(b) != q[10]:
                print('pso %x: code not captured' % (w & 0xffffff))
                continue
            key = fnv(root, fnv(b, fnv(a)))
            for ext, blob in (('.a', a), ('.b', b), ('.root', root)):
                open(os.path.join(out_dir, 'req_%016x%s' % (key, ext)), 'wb').write(blob)
            print('pso %x -> request %016x' % (w & 0xffffff, key))
    # translate every request that has no result yet
    for name in sorted(os.listdir(out_dir)):
        m = re.fullmatch(r'req_([0-9a-f]{16})\.a', name)
        if not m:
            continue
        key = int(m.group(1), 16)
        if os.path.exists(os.path.join(out_dir, '%016x.txt' % key)):
            continue
        print('%016x: %s' % (key, translate_request(out_dir, key)))


if __name__ == '__main__':
    main()
