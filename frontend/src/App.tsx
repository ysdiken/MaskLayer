import { useState, useRef, useCallback, useEffect, useLayoutEffect, useMemo } from 'react'
import {
  type MaskResponse, type UploadMaskResponse, type ManualSpanRequest, type Span,
  type UnmaskResponse, type PageExtractionInfo,
  labelColour, TAXONOMY_LABELS,
} from './types'

const API = 'http://localhost:8000'

// ── Page splitting utilities ──────────────────────────────────────────────────

interface PageSlice {
  text: string        // raw text of this page
  startOffset: number // char offset into the full text
  pageIndex: number   // 0-based
}

/**
 * Split text into pages.
 *
 * If the text was produced by the upload endpoint it contains \f characters
 * at real page boundaries. Otherwise we auto-paginate at double-newlines,
 * grouping paragraphs until a page reaches ~1 200 chars.
 */
function splitIntoPages(text: string, pageSep: string | null): PageSlice[] {
  if (!text) return []

  // Real page boundaries from the upload endpoint
  if (pageSep && text.includes(pageSep)) {
    const parts = text.split(pageSep)
    const pages: PageSlice[] = []
    let offset = 0
    parts.forEach((t, i) => {
      pages.push({ text: t, startOffset: offset, pageIndex: i })
      offset += t.length + pageSep.length
    })
    return pages
  }

  // Auto-paginate plain text: group consecutive paragraphs up to ~1 200 chars
  const MAX_PAGE = 1200
  const paras = text.split(/\n\n+/)
  const pages: PageSlice[] = []
  let curText = ''
  let curStart = 0
  let offset = 0

  for (const para of paras) {
    const blockLen = para.length + 2          // +2 for the \n\n we split on
    if (curText && (curText.length + blockLen) > MAX_PAGE) {
      pages.push({ text: curText, startOffset: curStart, pageIndex: pages.length })
      curStart = offset
      curText  = para
    } else {
      curText = curText ? curText + '\n\n' + para : para
    }
    offset += blockLen
  }
  if (curText) pages.push({ text: curText, startOffset: curStart, pageIndex: pages.length })
  return pages
}

// ── Page card wrapper ─────────────────────────────────────────────────────────

function PageCard({
  pageIndex, totalPages, extractionInfo, children,
}: {
  pageIndex: number
  totalPages: number
  extractionInfo?: PageExtractionInfo
  children: React.ReactNode
}) {
  const method = extractionInfo?.method
  const conf   = extractionInfo?.confidence

  return (
    <div className="relative bg-slate-800 rounded-sm shadow-[0_4px_20px_rgba(0,0,0,0.45)] border border-slate-700 mx-auto"
         style={{ width: '100%', maxWidth: '720px' }}>
      {/* Page top bar */}
      <div className="flex items-center justify-between px-4 py-1.5 border-b border-slate-700 bg-slate-900/50 rounded-t-sm">
        <span className="text-[10px] text-slate-500 font-mono tabular-nums">
          Sayfa {pageIndex + 1} / {totalPages}
        </span>
        {method && (
          <span className={`text-[10px] font-medium rounded-full px-2 py-0.5 border ${
            method === 'ocr'
              ? 'bg-amber-50 text-amber-700 border-amber-200'
              : method === 'docx'
              ? 'bg-blue-50 text-blue-700 border-blue-200'
              : 'bg-green-50 text-green-600 border-green-200'
          }`}>
            {method === 'ocr'
              ? `OCR${conf !== null && conf !== undefined ? ` · ${Math.round(conf)}%` : ''}`
              : method === 'docx' ? 'DOCX' : 'Native PDF'}
          </span>
        )}
      </div>

      {/* Page body — document-like padding mimics real page margins */}
      <div className="px-10 py-8">
        {children}
      </div>
    </div>
  )
}

// ── Masked text renderer (right panel) — paged ───────────────────────────────

function MaskedTextView({
  text, spans, pageSep, extraction, currentPage,
}: {
  text: string
  spans: Span[]
  pageSep: string | null
  extraction?: PageExtractionInfo[]
  currentPage: number
}) {
  const pages = useMemo(() => splitIntoPages(text, pageSep), [text, pageSep])

  // Build placeholder → label map once
  const placeholderLabel = useMemo(() => {
    const map: Record<string, string> = {}
    for (const s of spans) {
      const re = new RegExp(`\\{${s.label}_\\d+\\}`, 'g')
      let m: RegExpExecArray | null
      while ((m = re.exec(text)) !== null) map[m[0]] = s.label
    }
    return map
  }, [text, spans])

  function renderPageText(pageText: string) {
    const parts = pageText.split(/(\{[A-Za-z_]+_\d+\})/g)
    return (
      <div className="whitespace-pre-wrap text-sm leading-7 text-slate-200 font-serif">
        {parts.map((part, i) => {
          const label = placeholderLabel[part]
          if (!label) return <span key={i}>{part}</span>
          const c = labelColour(label)
          return (
            <span key={i}
              className={`inline-flex items-center rounded px-1.5 py-0.5 text-xs font-semibold border mx-0.5 ${c.bg} ${c.text} ${c.border}`}
              title={label}>
              {part}
            </span>
          )
        })}
      </div>
    )
  }

  const page = pages[currentPage] ?? pages[0]

  return (
    <PageCard
      pageIndex={currentPage}
      totalPages={pages.length}
      extractionInfo={extraction?.[currentPage]}
    >
      {renderPageText(page?.text ?? text)}
    </PageCard>
  )
}

// ── Annotated original text (left panel in review mode) ───────────────────────
// Renders already-detected spans as coloured chips. Plain text between
// chips is normal, selectable text. Selecting any plain text and releasing
// the mouse shows the manual-mask popover.

interface Segment {
  type: 'plain' | 'span'
  text: string
  span?: Span
}

function buildSegments(text: string, spans: Span[]): Segment[] {
  const sorted = [...spans].sort((a, b) => a.start - b.start)
  const segs: Segment[] = []
  let cursor = 0
  for (const s of sorted) {
    if (s.start > cursor) segs.push({ type: 'plain', text: text.slice(cursor, s.start) })
    segs.push({ type: 'span', text: text.slice(s.start, s.end), span: s })
    cursor = s.end
  }
  if (cursor < text.length) segs.push({ type: 'plain', text: text.slice(cursor) })
  return segs
}

function AnnotatedText({
  text, spans, pageSep, extraction, currentPage,
  onAddSpan,
}: {
  text: string
  spans: Span[]
  pageSep: string | null
  extraction?: PageExtractionInfo[]
  currentPage: number
  onAddSpan: (selected: string, x: number, y: number, yAbove: number) => void
}) {
  const pages = useMemo(() => splitIntoPages(text, pageSep), [text, pageSep])

  function handleMouseUp(e: React.MouseEvent) {
    e.stopPropagation()
    const selection = window.getSelection()
    if (!selection || selection.isCollapsed) return
    const selected = selection.toString()
    if (!selected.trim()) return
    const range = selection.getRangeAt(0)
    const rect  = range.getBoundingClientRect()
    const x = Math.min(Math.max(8, rect.left + rect.width / 2 - 160), window.innerWidth - 328)
    const y = rect.bottom + 6
    // Top edge of the selection — lets the popover flip above it when there
    // is not enough room below (selection near the bottom of the window).
    const yAbove = rect.top - 6
    onAddSpan(selected.trim(), x, y, yAbove)
  }

  function renderPageSegments(page: PageSlice) {
    // Filter spans that fall within this page, rebase their offsets
    const pageSpans = spans
      .filter(s => s.start >= page.startOffset && s.end <= page.startOffset + page.text.length)
      .map(s => ({ ...s, start: s.start - page.startOffset, end: s.end - page.startOffset }))
    const segs = buildSegments(page.text, pageSpans)

    return segs.map((seg, i) => {
      if (seg.type === 'plain') {
        return (
          <span key={i}>
            {seg.text.split('\n').map((line, j, arr) => (
              <span key={j}>{line}{j < arr.length - 1 ? <br /> : null}</span>
            ))}
          </span>
        )
      }
      const s   = seg.span!
      const c   = labelColour(s.label)
      const src = s.source === 'manual' ? '✎' : s.source === 'ner' ? '◉' : s.source === 'gazetteer' ? '▣' : '◆'
      return (
        <span key={i}
          className={`inline-flex items-center gap-0.5 rounded px-1.5 py-0.5 text-xs font-semibold border mx-0.5 ${c.bg} ${c.text} ${c.border}`}
          title={`${s.label} [${s.source}] conf=${(s.confidence * 100).toFixed(0)}%`}>
          <span className="opacity-50 text-[9px]">{src}</span>
          {seg.text}
        </span>
      )
    })
  }

  const page = pages[currentPage] ?? pages[0]
  if (!page) return null

  return (
    <div
      onMouseUp={handleMouseUp}
      className="flex-1 overflow-auto p-4 select-text cursor-text"
    >
      <PageCard pageIndex={currentPage} totalPages={pages.length} extractionInfo={extraction?.[currentPage]}>
        <div className="text-sm font-serif leading-7 text-slate-200">
          {renderPageSegments(page)}
        </div>
      </PageCard>
    </div>
  )
}

// ── Label picker popover ──────────────────────────────────────────────────────

interface PopoverState {
  x: number
  y: number       // preferred position: just below the selection
  yAbove: number  // fallback anchor: just above the selection (flip target)
  selectedText: string
}

function LabelPopover({
  popover, onConfirm, onClose,
}: {
  popover: PopoverState
  onConfirm: (text: string, label: string) => void
  onClose: () => void
}) {
  const [picked,    setPicked]    = useState<string | null>(null)
  const [custom,    setCustom]    = useState('')
  const [useCustom, setUseCustom] = useState(false)
  const [top,       setTop]       = useState(popover.y)
  const ref = useRef<HTMLDivElement>(null)

  // Keep the popover fully inside the viewport: place it below the selection
  // when it fits, otherwise flip it above. Runs before paint (no flicker)
  // and re-runs when content height changes (e.g. custom-label validation).
  useLayoutEffect(() => {
    const h = ref.current?.offsetHeight ?? 0
    if (popover.y + h > window.innerHeight - 8) {
      setTop(Math.max(8, popover.yAbove - h))
    } else {
      setTop(popover.y)
    }
  }, [popover, picked, useCustom])

  // Close on mousedown outside — not onClick, to avoid race with mouseup
  useEffect(() => {
    function onMouseDown(e: MouseEvent) {
      if (ref.current && !ref.current.contains(e.target as Node)) onClose()
    }
    // slight delay so the mouseup that opened us doesn't immediately close us
    const tid = window.setTimeout(() =>
      document.addEventListener('mousedown', onMouseDown), 50)
    return () => {
      clearTimeout(tid)
      document.removeEventListener('mousedown', onMouseDown)
    }
  }, [onClose])

  // Close on Escape
  useEffect(() => {
    function onKey(e: KeyboardEvent) { if (e.key === 'Escape') onClose() }
    document.addEventListener('keydown', onKey)
    return () => document.removeEventListener('keydown', onKey)
  }, [onClose])

  const effectiveLabel = useCustom ? custom.trim() : picked

  return (
    <div
      ref={ref}
      style={{ position: 'fixed', left: popover.x, top, zIndex: 9999 }}
      className="w-80 max-h-[calc(100vh-16px)] overflow-y-auto rounded-xl border border-slate-600 bg-slate-800 shadow-2xl p-4 flex flex-col gap-3"
      onMouseDown={e => e.stopPropagation()}  // prevent self-closing
    >
      {/* Selected text preview */}
      <div>
        <p className="text-[10px] font-semibold text-gray-400 uppercase tracking-wide mb-1">Seçilen metin</p>
        <div className="rounded bg-yellow-50 border border-yellow-200 px-2 py-1.5 text-sm font-mono text-yellow-900 truncate">
          "{popover.selectedText}"
        </div>
      </div>

      {/* Label grid */}
      <div>
        <p className="text-[10px] font-semibold text-gray-400 uppercase tracking-wide mb-2">Tür seçin</p>
        <div className="grid grid-cols-3 gap-1">
          {TAXONOMY_LABELS.map(({ label, display }) => {
            const c      = labelColour(label)
            const active = picked === label && !useCustom
            return (
              <button key={label}
                onMouseDown={e => e.stopPropagation()}
                onClick={() => { setPicked(label); setUseCustom(false) }}
                className={`rounded px-2 py-1.5 text-[11px] font-medium border text-left transition-all ${
                  active
                    ? `${c.bg} ${c.text} ${c.border} ring-2 ring-offset-1 ring-blue-400`
                    : `${c.bg} ${c.text} ${c.border} hover:opacity-80`
                }`}>
                {display}
              </button>
            )
          })}
        </div>
      </div>

      {/* Custom label */}
      <div>
        <p className="text-[10px] font-semibold text-gray-400 uppercase tracking-wide mb-1">Özel tür</p>
        <input
          type="text"
          placeholder="örn. Company_Secret"
          value={custom}
          onMouseDown={e => e.stopPropagation()}
          onChange={e => { setCustom(e.target.value); setUseCustom(true); setPicked(null) }}
          onFocus={() => { setUseCustom(true); setPicked(null) }}
          className={`w-full rounded border bg-slate-900 text-slate-100 placeholder:text-slate-500 px-2 py-1.5 text-xs focus:outline-none focus:ring-2 focus:ring-blue-400 ${
            useCustom ? 'border-blue-400 ring-1 ring-blue-400/40' : 'border-slate-600'
          }`}
        />
      </div>

      <p className="text-[10px] text-slate-400">
        Metindeki tüm "<span className="font-mono">{popover.selectedText}</span>" tekrarları maskelenecek.
      </p>

      {/* Actions */}
      <div className="flex gap-2">
        <button onClick={onClose} onMouseDown={e => e.stopPropagation()}
          className="flex-1 rounded-lg border border-slate-600 py-2 text-xs text-slate-300 hover:bg-slate-700 transition-colors">
          İptal
        </button>
        <button
          onMouseDown={e => e.stopPropagation()}
          onClick={() => effectiveLabel && onConfirm(popover.selectedText, effectiveLabel)}
          disabled={!effectiveLabel}
          className="flex-1 rounded-lg bg-blue-600 py-2 text-xs font-semibold text-white hover:bg-blue-700 disabled:opacity-40 transition-colors">
          Tümünü Maskele
        </button>
      </div>
    </div>
  )
}

// ── Span list sidebar ─────────────────────────────────────────────────────────

function SpanList({
  spans,
  onRemove,
  onFlag,
  flagged,
  disabledLabels,
}: {
  spans: Span[]
  onRemove?: (s: Span) => void
  onFlag?: (s: Span) => void
  flagged?: Set<string>
  disabledLabels?: Set<string>
}) {
  const [activeLabels,  setActiveLabels]  = useState<Set<string>>(new Set())
  const [activeSources, setActiveSources] = useState<Set<string>>(new Set())
  const [flaggedOnly,   setFlaggedOnly]   = useState(false)

  // Unique labels and sources present in this span list
  const allLabels  = useMemo(() => [...new Set(spans.map(s => s.label))].sort(), [spans])
  const allSources = useMemo(() => [...new Set(spans.map(s => s.source))].sort(), [spans])

  function toggleLabel(label: string) {
    setActiveLabels(prev => {
      const next = new Set(prev)
      next.has(label) ? next.delete(label) : next.add(label)
      return next
    })
  }

  function toggleSource(src: string) {
    setActiveSources(prev => {
      const next = new Set(prev)
      next.has(src) ? next.delete(src) : next.add(src)
      return next
    })
  }

  const filtered = useMemo(() => {
    return [...spans]
      .sort((a, b) => a.start - b.start)
      .filter(s => activeLabels.size  === 0 || activeLabels.has(s.label))
      .filter(s => activeSources.size === 0 || activeSources.has(s.source))
      .filter(s => {
        if (!flaggedOnly) return true
        const key = `${s.start}:${s.end}:${s.label}`
        return flagged?.has(key) ?? false
      })
  }, [spans, activeLabels, activeSources, flaggedOnly, flagged])

  const srcLabel: Record<string, string> = { regex: 'Regex', ner: 'NER', manual: 'Elle', gazetteer: 'Sözlük' }
  const srcBadgeBase: Record<string, string> = {
    manual:    'bg-orange-100 text-orange-700 border-orange-200',
    ner:       'bg-purple-100 text-purple-700 border-purple-200',
    regex:     'bg-blue-100   text-blue-700   border-blue-200',
    gazetteer: 'bg-teal-100   text-teal-700   border-teal-200',
  }
  const srcBadgeActive: Record<string, string> = {
    manual:    'bg-orange-500 text-white border-orange-500',
    ner:       'bg-purple-500 text-white border-purple-500',
    regex:     'bg-blue-500   text-white border-blue-500',
    gazetteer: 'bg-teal-500   text-white border-teal-500',
  }

  const flagCount = flagged
    ? spans.filter(s => flagged.has(`${s.start}:${s.end}:${s.label}`)).length
    : 0

  return (
    <div className="flex flex-col gap-2">

      {/* ── Filter bar ── */}
      <div className="flex flex-col gap-2 bg-slate-800/60 rounded-xl border border-slate-700 p-2.5">

        {/* Source toggles */}
        <div className="flex items-center gap-1.5 flex-wrap">
          <span className="text-[10px] text-slate-500 font-medium uppercase tracking-wide mr-0.5">Kaynak</span>
          {allSources.map(src => (
            <button key={src}
              onClick={() => toggleSource(src)}
              className={`rounded-full border px-2 py-0.5 text-[10px] font-medium transition-colors ${
                activeSources.has(src) ? srcBadgeActive[src] : srcBadgeBase[src]
              }`}>
              {srcLabel[src] ?? src} ({spans.filter(s => s.source === src).length})
            </button>
          ))}
        </div>

        {/* Label pills */}
        <div className="flex items-center gap-1.5 flex-wrap">
          <span className="text-[10px] text-slate-500 font-medium uppercase tracking-wide mr-0.5">Tür</span>
          {allLabels.map(label => {
            const c       = labelColour(label)
            const isActive = activeLabels.has(label)
            const cnt     = spans.filter(s => s.label === label).length
            return (
              <button key={label} onClick={() => toggleLabel(label)}
                className={`rounded-full border px-2 py-0.5 text-[10px] font-medium transition-colors ${
                  isActive
                    ? `${c.bg} ${c.text} ${c.border} ring-1 ring-offset-0 ring-current`
                    : 'bg-slate-900 text-slate-400 border-slate-700 hover:border-slate-500'
                }`}>
                {label} {cnt > 1 ? `(${cnt})` : ''}
              </button>
            )
          })}
        </div>

        {/* Flagged-only toggle + clear */}
        <div className="flex items-center justify-between">
          <button
            onClick={() => setFlaggedOnly(f => !f)}
            className={`rounded-full border px-2 py-0.5 text-[10px] font-medium transition-colors ${
              flaggedOnly
                ? 'bg-red-100 text-red-600 border-red-300'
                : 'bg-slate-900 text-slate-500 border-slate-700 hover:border-red-400 hover:text-red-400'
            }`}>
            ⚑ İşaretlenenler{flagCount > 0 ? ` (${flagCount})` : ''}
          </button>
          {(activeLabels.size > 0 || activeSources.size > 0 || flaggedOnly) && (
            <button
              onClick={() => { setActiveLabels(new Set()); setActiveSources(new Set()); setFlaggedOnly(false) }}
              className="text-[10px] text-slate-500 hover:text-slate-200 transition-colors">
              Filtreyi Temizle ×
            </button>
          )}
        </div>
      </div>

      {/* ── Result count ── */}
      {(activeLabels.size > 0 || activeSources.size > 0 || flaggedOnly) && (
        <p className="text-[10px] text-slate-500 px-0.5">
          {filtered.length} / {spans.length} varlık gösteriliyor
        </p>
      )}

      {/* ── Span cards ── */}
      {filtered.length === 0 && (
        <p className="text-xs text-slate-500 text-center py-4">Filtrelerle eşleşen varlık yok.</p>
      )}

      {filtered.map((s, i) => {
        const c         = labelColour(s.label)
        const confPct   = Math.round(s.confidence * 100)
        const confColor = s.confidence >= 0.85 ? 'bg-green-400' : s.confidence >= 0.60 ? 'bg-yellow-400' : 'bg-red-400'
        const srcBadge  =
          s.source === 'manual'    ? 'bg-orange-200 text-orange-800' :
          s.source === 'ner'       ? 'bg-purple-200 text-purple-800' :
          s.source === 'gazetteer' ? 'bg-teal-200   text-teal-800'   :
                                     'bg-blue-200   text-blue-800'
        const spanKey   = `${s.start}:${s.end}:${s.label}`
        const isFlagged = flagged?.has(spanKey) ?? false
        const policyOff = disabledLabels?.has(s.label) ?? false

        return (
          <div key={i} className={`rounded-lg border p-2.5 text-xs transition-opacity ${c.bg} ${c.border} ${isFlagged || policyOff ? 'opacity-50' : ''}`}>
            <div className="flex items-center justify-between mb-1">
              <span className={`font-bold ${c.text}`}>{s.label}</span>
              <div className="flex items-center gap-1">
                {/* Detected but not masked in the output (policy disabled) */}
                {policyOff && (
                  <span className="rounded-full px-1.5 py-0.5 text-[10px] font-medium bg-slate-200 text-slate-600 border border-slate-300"
                    title="Politika gereği çıktıda maskelenmiyor">
                    maskelenmiyor
                  </span>
                )}
                <span className={`rounded-full px-1.5 py-0.5 text-[10px] font-medium ${srcBadge}`}>
                  {s.source.toUpperCase()}
                </span>
                {/* Manual spans: hard remove (re-masks without the span) */}
                {s.source === 'manual' && onRemove && (
                  <button onClick={() => onRemove(s)} title="Kaldır"
                    className="text-red-400 hover:text-red-700 font-bold leading-none text-sm">×</button>
                )}
                {/* Auto spans: soft flag as false positive (logs to learning DB) */}
                {s.source !== 'manual' && onFlag && (
                  <button
                    onClick={() => !isFlagged && onFlag(s)}
                    title={isFlagged ? 'Yanlış pozitif olarak işaretlendi' : 'Yanlış tespit olarak işaretle'}
                    className={`text-[10px] leading-none rounded px-1 py-0.5 border transition-colors ${
                      isFlagged
                        ? 'bg-red-100 text-red-500 border-red-200 cursor-default'
                        : 'bg-white text-gray-400 border-gray-200 hover:bg-red-50 hover:text-red-500 hover:border-red-200'
                    }`}>
                    {isFlagged ? '⚑ işaretlendi' : '⚑'}
                  </button>
                )}
              </div>
            </div>
            <div className={`font-mono truncate mb-1.5 ${c.text}`} title={s.text}>{s.text}</div>
            <div className="flex items-center gap-2">
              <div className="flex-1 bg-white/60 rounded-full h-1.5">
                <div className={`h-1.5 rounded-full ${confColor}`} style={{ width: `${confPct}%` }} />
              </div>
              <span className="text-gray-500 tabular-nums">{confPct}%</span>
            </div>
            <div className="mt-1 text-gray-400 tabular-nums">[{s.start}:{s.end}]</div>
          </div>
        )
      })}
    </div>
  )
}

// ── Unmask panel ──────────────────────────────────────────────────────────────

function UnmaskPanel({ jobId }: { jobId: string }) {
  const [llmText, setLlmText] = useState('')
  const [result,  setResult]  = useState<string | null>(null)
  const [loading, setLoading] = useState(false)
  const [error,   setError]   = useState<string | null>(null)

  async function handleUnmask() {
    setLoading(true); setError(null); setResult(null)
    try {
      const res = await fetch(`${API}/api/v1/unmask`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ job_id: jobId, masked_text: llmText }),
      })
      if (!res.ok) { const e = await res.json(); throw new Error(e.detail ?? `${res.status}`) }
      const data: UnmaskResponse = await res.json()
      setResult(data.unmasked_text)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unknown error')
    } finally { setLoading(false) }
  }

  return (
    <div className="flex flex-col gap-3">
      <p className="text-xs text-slate-400 leading-relaxed">
        LLM yanıtını yapıştırın —{' '}
        <code className="bg-slate-800 text-slate-300 px-1 rounded">{'{placeholder}'}</code>{' '}
        içeren metindeki değerler sunucu tarafında geri yüklenir.
      </p>
      <textarea
        className="h-32 resize-none rounded-lg border border-slate-700 bg-slate-800 text-slate-100 placeholder:text-slate-500 p-3 text-sm font-mono focus:outline-none focus:ring-2 focus:ring-purple-400"
        placeholder="Müşteri {TC_No_1} için işlem onaylandı…"
        value={llmText}
        onChange={e => setLlmText(e.target.value)}
      />
      <button onClick={handleUnmask} disabled={loading || !llmText.trim()}
        className="rounded-lg bg-purple-600 px-4 py-2 text-sm font-semibold text-white hover:bg-purple-700 disabled:opacity-50 transition-colors">
        {loading ? 'Yükleniyor…' : 'Orijinal Değerleri Geri Yükle'}
      </button>
      {error  && <p className="text-xs text-red-600">{error}</p>}
      {result && (
        <div className="rounded-lg border border-green-800 bg-green-950/50 text-green-200 p-3 text-sm whitespace-pre-wrap font-mono">
          {result}
        </div>
      )}
    </div>
  )
}

// ── Extraction info badge (upload results only) ───────────────────────────────

// ── File drop zone ─────────────────────────────────────────────────────────────

function DropZone({
  onFile, loading,
}: {
  onFile: (f: File) => void
  loading: boolean
}) {
  const [dragOver, setDragOver] = useState(false)
  const inputRef = useRef<HTMLInputElement>(null)

  function handleDrop(e: React.DragEvent) {
    e.preventDefault(); setDragOver(false)
    const file = e.dataTransfer.files[0]
    if (file) onFile(file)
  }

  return (
    <div
      onDragOver={e => { e.preventDefault(); setDragOver(true) }}
      onDragLeave={() => setDragOver(false)}
      onDrop={handleDrop}
      onClick={() => !loading && inputRef.current?.click()}
      className={`flex-1 flex flex-col items-center justify-center gap-3 cursor-pointer transition-colors m-4 rounded-xl border-2 border-dashed
        ${dragOver  ? 'border-blue-400 bg-blue-950/40'  : 'border-slate-700 bg-slate-800/40 hover:border-blue-500 hover:bg-blue-950/30'}
        ${loading   ? 'opacity-50 cursor-not-allowed' : ''}
      `}
    >
      <input
        ref={inputRef}
        type="file"
        className="hidden"
        accept=".pdf,.docx,.doc,.png,.jpg,.jpeg,.tiff,.tif,.bmp,.webp"
        onChange={e => { const f = e.target.files?.[0]; if (f) onFile(f); e.target.value = '' }}
      />
      <div className="text-3xl">{loading ? '⏳' : '📂'}</div>
      <div className="text-center">
        <p className="text-sm font-medium text-slate-300">
          {loading ? 'Yükleniyor ve maskeleniyor…' : 'PDF, DOCX veya görsel sürükleyin'}
        </p>
        <p className="text-xs text-gray-400 mt-1">veya seçmek için tıklayın</p>
        <p className="text-[10px] text-gray-400 mt-0.5">.pdf · .docx · .png · .jpg · .tiff</p>
      </div>
    </div>
  )
}

// ── Legend ────────────────────────────────────────────────────────────────────

function Legend() {
  return (
    <div className="flex flex-wrap gap-1.5">
      {TAXONOMY_LABELS.map(({ label, display }) => {
        const c = labelColour(label)
        return (
          <span key={label}
            className={`inline-flex items-center rounded-full border px-2 py-0.5 text-[10px] font-medium ${c.bg} ${c.text} ${c.border}`}>
            {display}
          </span>
        )
      })}
    </div>
  )
}

// ── Ablation dashboard ────────────────────────────────────────────────────────

interface ModeResult {
  span_count: number
  processing_time_ms: number
  mean_confidence: number
  by_label: Record<string, number>
  spans: Span[]
}

interface AblationResult {
  regex: ModeResult
  ner:   ModeResult
  full:  ModeResult
  overlap: {
    regex_only_count: number
    ner_only_count:   number
    both_count:       number
    regex_only_spans: Span[]
    ner_only_spans:   Span[]
    both_spans:       Span[]
  }
}

function AblationPanel({ text }: { text: string }) {
  const [result,  setResult]  = useState<AblationResult | null>(null)
  const [loading, setLoading] = useState(false)
  const [error,   setError]   = useState<string | null>(null)

  async function run() {
    setLoading(true); setError(null); setResult(null)
    try {
      const res = await fetch(`${API}/api/v1/ablation`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text }),
      })
      if (!res.ok) { const e = await res.json(); throw new Error(e.detail ?? `${res.status}`) }
      setResult(await res.json())
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Unknown error')
    } finally { setLoading(false) }
  }

  function exportCSV() {
    if (!result) return
    // Collect all unique labels
    const allLabels = [...new Set([
      ...Object.keys(result.regex.by_label),
      ...Object.keys(result.ner.by_label),
      ...Object.keys(result.full.by_label),
    ])].sort()
    const rows = [
      ['Label', 'Regex', 'NER', 'Full (Regex+NER)', 'Regex Only', 'NER Only', 'Both'],
      ...allLabels.map(l => [
        l,
        result.regex.by_label[l] ?? 0,
        result.ner.by_label[l]   ?? 0,
        result.full.by_label[l]  ?? 0,
        result.overlap.regex_only_spans.filter(s => s.label === l).length,
        result.overlap.ner_only_spans.filter(s => s.label === l).length,
        result.overlap.both_spans.filter(s => s.label === l).length,
      ]),
      [],
      ['', 'Regex', 'NER', 'Full'],
      ['Span Count', result.regex.span_count, result.ner.span_count, result.full.span_count],
      ['Time (ms)', result.regex.processing_time_ms.toFixed(1), result.ner.processing_time_ms.toFixed(1), result.full.processing_time_ms.toFixed(1)],
      ['Mean Confidence', result.regex.mean_confidence.toFixed(3), result.ner.mean_confidence.toFixed(3), result.full.mean_confidence.toFixed(3)],
    ]
    const csv = rows.map(r => r.join(',')).join('\n')
    const blob = new Blob([csv], { type: 'text/csv' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a'); a.href = url
    a.download = `ablation-${Date.now()}.csv`; a.click()
    URL.revokeObjectURL(url)
  }

  const allLabels = result ? [...new Set([
    ...Object.keys(result.regex.by_label),
    ...Object.keys(result.ner.by_label),
    ...Object.keys(result.full.by_label),
  ])].sort() : []

  const maxCount = result
    ? Math.max(1, ...allLabels.map(l => result.full.by_label[l] ?? 0))
    : 1

  return (
    <div className="flex flex-col gap-5">

      {/* Run button */}
      <div className="flex items-center gap-3">
        <button onClick={run} disabled={loading || !text.trim()}
          className="flex-1 rounded-lg bg-indigo-600 py-2.5 text-sm font-semibold text-white hover:bg-indigo-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors flex items-center justify-center gap-2">
          {loading
            ? <><svg className="animate-spin h-4 w-4" viewBox="0 0 24 24" fill="none"><circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"/><path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8H4z"/></svg>Çalışıyor…</>
            : '▶ Ablasyon Analizini Çalıştır'}
        </button>
        {result && (
          <button onClick={exportCSV}
            className="rounded-lg border border-slate-700 bg-slate-800 px-3 py-2.5 text-xs font-medium text-slate-300 hover:bg-slate-700 transition-colors">
            ↓ CSV
          </button>
        )}
      </div>
      {error && <p className="text-xs text-red-600">{error}</p>}

      {!result && !loading && (
        <div className="flex flex-col items-center justify-center gap-3 text-slate-500 text-center py-10">
          <div className="w-12 h-12 rounded-xl bg-indigo-950/60 flex items-center justify-center text-2xl">📊</div>
          <div>
            <p className="text-sm font-medium text-slate-300">Ablasyon Analizi</p>
            <p className="text-xs mt-1">Aynı metni Regex, NER ve Tam mod ile çalıştırır.<br/>Her modun ne tespit ettiğini karşılaştırır.</p>
          </div>
        </div>
      )}

      {result && (
        <>
          {/* ── Mode summary cards ── */}
          <div className="grid grid-cols-3 gap-2">
            {([
              { key: 'regex', label: 'Regex',        color: 'text-blue-300',   bg: 'bg-blue-950/40',   border: 'border-blue-900'   },
              { key: 'ner',   label: 'NER',          color: 'text-purple-300', bg: 'bg-purple-950/40', border: 'border-purple-900' },
              { key: 'full',  label: 'Regex + NER',  color: 'text-indigo-300', bg: 'bg-indigo-950/40', border: 'border-indigo-900' },
            ] as const).map(({ key, label, color, bg, border }) => {
              const m = result[key]
              return (
                <div key={key} className={`rounded-xl border p-3 ${bg} ${border}`}>
                  <p className={`text-[10px] font-bold uppercase tracking-wide ${color} mb-2`}>{label}</p>
                  <p className={`text-2xl font-bold tabular-nums ${color}`}>{m.span_count}</p>
                  <p className="text-[10px] text-slate-400 mt-0.5">varlık</p>
                  <div className="mt-2 border-t border-slate-700/60 pt-2 flex flex-col gap-0.5">
                    <p className="text-[10px] text-slate-400 tabular-nums">{m.processing_time_ms.toFixed(0)} ms</p>
                    <p className="text-[10px] text-slate-400 tabular-nums">ort. %{Math.round(m.mean_confidence * 100)} güven</p>
                  </div>
                </div>
              )
            })}
          </div>

          {/* ── Overlap Venn summary ── */}
          <div className="rounded-xl border border-slate-700 bg-slate-800 p-4 shadow-sm">
            <p className="text-xs font-semibold text-slate-400 uppercase tracking-wide mb-3">Örtüşme Analizi</p>
            <div className="grid grid-cols-3 gap-2 text-center">
              {[
                { label: 'Yalnızca Regex', count: result.overlap.regex_only_count, color: 'text-blue-400',  bg: 'bg-blue-950/40'  },
                { label: 'Yalnızca NER',   count: result.overlap.ner_only_count,   color: 'text-purple-400', bg: 'bg-purple-950/40' },
                { label: 'Her İkisi',      count: result.overlap.both_count,       color: 'text-indigo-400', bg: 'bg-indigo-950/40' },
              ].map(({ label, count, color, bg }) => (
                <div key={label} className={`rounded-lg p-2 ${bg}`}>
                  <p className={`text-xl font-bold tabular-nums ${color}`}>{count}</p>
                  <p className="text-[10px] text-slate-400 mt-0.5 leading-tight">{label}</p>
                </div>
              ))}
            </div>
          </div>

          {/* ── Per-label breakdown ── */}
          {allLabels.length > 0 && (
            <div className="rounded-xl border border-slate-700 bg-slate-800 p-4 shadow-sm">
              <p className="text-xs font-semibold text-slate-400 uppercase tracking-wide mb-3">Etiket Bazlı Karşılaştırma</p>

              {/* Legend */}
              <div className="flex items-center gap-3 mb-3 text-[10px] text-slate-400">
                <span className="flex items-center gap-1"><span className="w-2.5 h-2.5 rounded-sm bg-blue-400 inline-block"/>Regex</span>
                <span className="flex items-center gap-1"><span className="w-2.5 h-2.5 rounded-sm bg-purple-400 inline-block"/>NER</span>
                <span className="flex items-center gap-1"><span className="w-2.5 h-2.5 rounded-sm bg-indigo-500 inline-block"/>Tam</span>
              </div>

              <div className="flex flex-col gap-2">
                {allLabels
                  .filter(l => (result.full.by_label[l] ?? 0) > 0 ||
                               (result.regex.by_label[l] ?? 0) > 0 ||
                               (result.ner.by_label[l] ?? 0) > 0)
                  .sort((a, b) => (result.full.by_label[b] ?? 0) - (result.full.by_label[a] ?? 0))
                  .map(label => {
                    const rCount = result.regex.by_label[label] ?? 0
                    const nCount = result.ner.by_label[label]   ?? 0
                    const fCount = result.full.by_label[label]  ?? 0
                    return (
                      <div key={label}>
                        <div className="flex items-center justify-between text-xs mb-1">
                          <span className="font-medium text-slate-200 w-28 truncate">{label}</span>
                          <span className="tabular-nums text-slate-500 text-[10px]">
                            <span className="text-blue-500">{rCount}R</span>
                            {' · '}
                            <span className="text-purple-500">{nCount}N</span>
                            {' · '}
                            <span className="text-indigo-600 font-medium">{fCount}F</span>
                          </span>
                        </div>
                        {/* Stacked mini bars */}
                        <div className="flex gap-0.5 h-2">
                          <div className="bg-blue-300 rounded-sm"   style={{ width: `${(rCount / maxCount) * 100}%` }} title={`Regex: ${rCount}`} />
                          <div className="bg-purple-300 rounded-sm" style={{ width: `${(nCount / maxCount) * 100}%` }} title={`NER: ${nCount}`} />
                          <div className="bg-indigo-400 rounded-sm" style={{ width: `${(fCount / maxCount) * 100}%` }} title={`Full: ${fCount}`} />
                        </div>
                      </div>
                    )
                  })}
              </div>
            </div>
          )}

          {/* ── Exclusive detection lists ── */}
          {(result.overlap.ner_only_spans.length > 0 || result.overlap.regex_only_spans.length > 0) && (
            <div className="rounded-xl border border-slate-700 bg-slate-800 p-4 shadow-sm">
              <p className="text-xs font-semibold text-slate-400 uppercase tracking-wide mb-3">NER'in Kazandığı Tespitler</p>
              {result.overlap.ner_only_spans.length === 0
                ? <p className="text-xs text-gray-400">Yok</p>
                : <div className="flex flex-wrap gap-1.5">
                    {result.overlap.ner_only_spans.map((s, i) => (
                      <span key={i} className="rounded-full bg-purple-100 text-purple-700 border border-purple-200 px-2 py-0.5 text-[10px] font-medium"
                        title={`${s.label} — %${Math.round(s.confidence*100)} güven`}>
                        {s.text}
                      </span>
                    ))}
                  </div>
              }
              {result.overlap.regex_only_spans.length > 0 && (
                <>
                  <p className="text-xs font-semibold text-slate-400 uppercase tracking-wide mt-4 mb-3">Regex'in Kazandığı Tespitler</p>
                  <div className="flex flex-wrap gap-1.5">
                    {result.overlap.regex_only_spans.map((s, i) => (
                      <span key={i} className="rounded-full bg-blue-100 text-blue-700 border border-blue-200 px-2 py-0.5 text-[10px] font-medium"
                        title={`${s.label} — %${Math.round(s.confidence*100)} güven`}>
                        {s.text}
                      </span>
                    ))}
                  </div>
                </>
              )}
            </div>
          )}
        </>
      )}
    </div>
  )
}

// ── Learning stats panel ──────────────────────────────────────────────────────

interface ThresholdHint {
  count: number
  min: number
  p50: number
  p90: number
  max: number
  suggested_threshold: number
}

interface RepeatedFP {
  text_hash: string
  entity_label: string
  detector_source: string | null
  count: number
}

interface CorrectionStats {
  total: number
  by_label: Record<string, { false_positive: number; false_negative: number; relabel: number }>
  by_source: Record<string, { false_positive: number; false_negative: number; relabel: number }>
  recent: Array<{
    occurred_at: string
    event_type: string
    entity_label: string
    corrected_label: string | null
    detector_source: string | null
    confidence: number | null
  }>
  threshold_hints: Record<string, ThresholdHint>
  repeated_fp: RepeatedFP[]
}

function LearningPanel() {
  const [stats,   setStats]   = useState<CorrectionStats | null>(null)
  const [loading, setLoading] = useState(true)
  const [error,   setError]   = useState<string | null>(null)

  useEffect(() => {
    fetch(`${API}/api/v1/corrections/stats`)
      .then(r => r.json())
      .then(d => { setStats(d); setLoading(false) })
      .catch(e => { setError(e.message); setLoading(false) })
  }, [])

  function exportDataset() {
    if (!stats) return
    const blob = new Blob([JSON.stringify(stats, null, 2)], { type: 'application/json' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url; a.download = `masklayer-corrections-${Date.now()}.json`; a.click()
    URL.revokeObjectURL(url)
  }

  if (loading) return <div className="flex items-center justify-center h-24 text-gray-400 text-sm">Yükleniyor…</div>
  if (error)   return <div className="text-xs text-red-500 p-2">Hata: {error}</div>
  if (!stats)  return null

  const labels   = Object.entries(stats.by_label)
  const sources  = Object.entries(stats.by_source)
  const maxFP    = Math.max(1, ...labels.map(([, v]) => v.false_positive))

  const eventLabel: Record<string, string> = {
    false_positive: 'Yanlış Pozitif',
    false_negative: 'Eksik Tespit',
    relabel:        'Yeniden Etiket',
  }
  const eventColor: Record<string, string> = {
    false_positive: 'text-red-400',
    false_negative: 'text-amber-400',
    relabel:        'text-blue-400',
  }

  return (
    <div className="flex flex-col gap-5">

      {/* Summary cards */}
      <div className="grid grid-cols-3 gap-2">
        {[
          { label: 'Toplam Düzeltme', value: stats.total, color: 'text-slate-100' },
          { label: 'Yanlış Pozitif',  value: Object.values(stats.by_source).reduce((s, v) => s + (v.false_positive ?? 0), 0), color: 'text-red-600' },
          { label: 'Eksik Tespit',    value: Object.values(stats.by_source).reduce((s, v) => s + (v.false_negative ?? 0), 0), color: 'text-amber-600' },
        ].map(({ label, value, color }) => (
          <div key={label} className="rounded-lg border border-slate-700 bg-slate-800 p-3 text-center shadow-sm">
            <div className={`text-2xl font-bold tabular-nums ${color}`}>{value}</div>
            <div className="text-[10px] text-gray-400 mt-0.5 leading-tight">{label}</div>
          </div>
        ))}
      </div>

      {/* Per-label breakdown */}
      {labels.length > 0 && (
        <div className="rounded-xl border border-slate-700 bg-slate-800 p-4 shadow-sm">
          <p className="text-xs font-semibold text-slate-400 uppercase tracking-wide mb-3">Etiket Bazlı Düzeltmeler</p>
          <div className="flex flex-col gap-2">
            {labels.sort((a, b) => (b[1].false_positive + b[1].false_negative) - (a[1].false_positive + a[1].false_negative)).map(([label, v]) => (
              <div key={label}>
                <div className="flex items-center justify-between text-xs mb-0.5">
                  <span className="font-medium text-slate-200">{label}</span>
                  <span className="text-gray-400 tabular-nums">
                    <span className="text-red-500">{v.false_positive} FP</span>
                    {' · '}
                    <span className="text-amber-500">{v.false_negative} FN</span>
                    {v.relabel > 0 && <> · <span className="text-blue-500">{v.relabel} RL</span></>}
                  </span>
                </div>
                {/* FP bar */}
                <div className="h-1.5 bg-slate-700 rounded-full overflow-hidden">
                  <div className="h-full bg-red-400 rounded-full" style={{ width: `${(v.false_positive / maxFP) * 100}%` }} />
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Per-source precision estimate */}
      {sources.length > 0 && (
        <div className="rounded-xl border border-slate-700 bg-slate-800 p-4 shadow-sm">
          <p className="text-xs font-semibold text-slate-400 uppercase tracking-wide mb-3">Algılayıcı Kalitesi</p>
          <div className="flex flex-col gap-3">
            {sources.map(([src, v]) => {
              const fp = v.false_positive ?? 0
              const fn = v.false_negative ?? 0
              return (
                <div key={src} className="flex items-center gap-3">
                  <span className={`rounded-full px-2 py-0.5 text-[10px] font-medium w-14 text-center ${
                    src === 'ner' ? 'bg-purple-100 text-purple-700' : 'bg-blue-100 text-blue-700'
                  }`}>{src.toUpperCase()}</span>
                  <div className="flex-1 text-xs text-slate-400">
                    <span className="text-red-500">{fp} FP</span>
                    {' · '}
                    <span className="text-amber-500">{fn} FN</span>
                  </div>
                </div>
              )
            })}
          </div>
        </div>
      )}

      {/* Threshold hints from FP confidence distribution */}
      {Object.keys(stats.threshold_hints).length > 0 && (
        <div className="rounded-xl border border-slate-700 bg-slate-800 p-4 shadow-sm">
          <p className="text-xs font-semibold text-slate-400 uppercase tracking-wide mb-1">Eşik Önerileri</p>
          <p className="text-[10px] text-gray-400 mb-3">
            Yanlış pozitif güven dağılımından hesaplanmıştır. Eşiği P90 değerinin üzerine çıkarmak FP'lerin ~%%90'ını ortadan kaldırır.
          </p>
          <div className="flex flex-col gap-2">
            {Object.entries(stats.threshold_hints)
              .sort((a, b) => b[1].count - a[1].count)
              .map(([key, hint]) => {
                const [label, src] = key.split('::')
                return (
                  <div key={key} className="text-xs">
                    <div className="flex items-center justify-between mb-0.5">
                      <span className="font-medium text-slate-200">{label}</span>
                      <div className="flex items-center gap-1.5">
                        <span className="text-gray-400 text-[10px]">{hint.count} FP · P90={hint.p90}</span>
                        <span className="rounded-full bg-amber-100 text-amber-700 border border-amber-200 px-1.5 py-0.5 text-[10px] font-medium">
                          öneri ≥ {hint.suggested_threshold}
                        </span>
                        {src !== 'any' && (
                          <span className={`rounded-full px-1.5 py-0.5 text-[10px] font-medium ${
                            src === 'ner' ? 'bg-purple-100 text-purple-700' : 'bg-blue-100 text-blue-700'
                          }`}>{src}</span>
                        )}
                      </div>
                    </div>
                    {/* Mini distribution bar: min → P50 → P90 → max */}
                    <div className="relative h-2 bg-slate-700 rounded-full overflow-hidden">
                      <div className="absolute h-full bg-red-900/70 rounded-full" style={{ left: `${hint.min * 100}%`, width: `${(hint.max - hint.min) * 100}%` }} />
                      <div className="absolute h-full w-0.5 bg-amber-500" style={{ left: `${hint.p50 * 100}%` }} title={`P50: ${hint.p50}`} />
                      <div className="absolute h-full w-0.5 bg-red-500" style={{ left: `${hint.p90 * 100}%` }} title={`P90: ${hint.p90}`} />
                    </div>
                    <div className="flex justify-between text-[9px] text-slate-500 mt-0.5">
                      <span>{hint.min}</span><span>P50={hint.p50}</span><span>P90={hint.p90}</span><span>{hint.max}</span>
                    </div>
                  </div>
                )
              })}
          </div>
        </div>
      )}

      {/* Repeated FP patterns — same entity text flagged multiple times */}
      {stats.repeated_fp.length > 0 && (
        <div className="rounded-xl border border-amber-900 bg-amber-950/40 p-4 shadow-sm">
          <p className="text-xs font-semibold text-amber-400 uppercase tracking-wide mb-1">Tekrarlayan Yanlış Tespitler</p>
          <p className="text-[10px] text-amber-500 mb-3">
            Aynı varlık metni birden fazla kez yanlış pozitif olarak işaretlendi. Blocklist'e ekleme adayları.
          </p>
          <div className="flex flex-col gap-1.5">
            {stats.repeated_fp.map((fp, i) => (
              <div key={i} className="flex items-center gap-2 text-xs py-1 border-b border-amber-900/50 last:border-0">
                <span className="font-bold text-amber-400 tabular-nums w-6 text-center">{fp.count}×</span>
                <span className="font-medium text-slate-200">{fp.entity_label}</span>
                {fp.detector_source && (
                  <span className={`rounded-full px-1.5 py-0.5 text-[10px] font-medium ${
                    fp.detector_source === 'ner' ? 'bg-purple-100 text-purple-700' : 'bg-blue-100 text-blue-700'
                  }`}>{fp.detector_source}</span>
                )}
                <span className="ml-auto font-mono text-[9px] text-slate-500 truncate max-w-[100px]" title="SHA-256 metin hash'i">
                  {fp.text_hash.slice(0, 12)}…
                </span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Recent corrections feed */}
      {stats.recent.length > 0 && (
        <div className="rounded-xl border border-slate-700 bg-slate-800 p-4 shadow-sm">
          <p className="text-xs font-semibold text-slate-400 uppercase tracking-wide mb-3">Son Düzeltmeler</p>
          <div className="flex flex-col gap-1.5">
            {stats.recent.map((r, i) => (
              <div key={i} className="flex items-center gap-2 text-xs text-slate-300 py-1 border-b border-slate-700/50 last:border-0">
                <span className={`font-medium w-28 shrink-0 ${eventColor[r.event_type] ?? ''}`}>
                  {eventLabel[r.event_type] ?? r.event_type}
                </span>
                <span className="font-medium text-slate-200">{r.entity_label}</span>
                {r.corrected_label && <span className="text-slate-500">→ {r.corrected_label}</span>}
                {r.detector_source && (
                  <span className={`ml-auto rounded-full px-1.5 py-0.5 text-[10px] ${
                    r.detector_source === 'ner' ? 'bg-purple-100 text-purple-600' : 'bg-blue-100 text-blue-600'
                  }`}>{r.detector_source}</span>
                )}
              </div>
            ))}
          </div>
        </div>
      )}

      {stats.total === 0 && (
        <div className="flex flex-col items-center justify-center gap-3 text-slate-500 text-center py-8">
          <div className="w-10 h-10 rounded-xl bg-slate-800 flex items-center justify-center text-xl">🎓</div>
          <p className="text-xs">Henüz düzeltme yok.<br/>Varlıklar panelinde ⚑ ile yanlış tespitleri işaretleyin.</p>
        </div>
      )}

      <button onClick={exportDataset} disabled={stats.total === 0}
        className="rounded-lg border border-slate-700 bg-slate-800 px-4 py-2 text-xs font-medium text-slate-300 hover:bg-slate-700 disabled:opacity-40 disabled:cursor-not-allowed transition-colors">
        ↓ Veri Setini Dışa Aktar (JSON)
      </button>
    </div>
  )
}

// ── Admin / mask-policy panel ─────────────────────────────────────────────────
// A global OUTPUT filter: which entity types are actually rendered as
// placeholders. Detection, the model, and the corrections/training log are
// never affected — only the masked output and the Redis mapping change.

interface MaskLabelInfo { label: string; display: string; group: string }
interface MaskPolicyData { labels: MaskLabelInfo[]; policy: Record<string, boolean> }

function AdminPanel({
  onClose, onSaved,
}: {
  onClose: () => void
  onSaved: (policy: Record<string, boolean>) => void
}) {
  const [data,   setData]   = useState<MaskPolicyData | null>(null)
  const [draft,  setDraft]  = useState<Record<string, boolean>>({})
  const [saving, setSaving] = useState(false)
  const [error,  setError]  = useState<string | null>(null)

  useEffect(() => {
    fetch(`${API}/api/v1/admin/mask-policy`)
      .then(r => r.json())
      .then((d: MaskPolicyData) => { setData(d); setDraft(d.policy) })
      .catch(e => setError(e instanceof Error ? e.message : 'Yüklenemedi'))
  }, [])

  // Close on Escape
  useEffect(() => {
    function onKey(e: KeyboardEvent) { if (e.key === 'Escape') onClose() }
    document.addEventListener('keydown', onKey)
    return () => document.removeEventListener('keydown', onKey)
  }, [onClose])

  function toggle(label: string) {
    setDraft(d => ({ ...d, [label]: !d[label] }))
  }
  function setAll(value: boolean) {
    if (!data) return
    setDraft(Object.fromEntries(data.labels.map(l => [l.label, value])))
  }

  async function save() {
    setSaving(true); setError(null)
    try {
      const res = await fetch(`${API}/api/v1/admin/mask-policy`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ policy: draft }),
      })
      if (!res.ok) throw new Error(`${res.status}`)
      const updated: MaskPolicyData = await res.json()
      onSaved(updated.policy)
      onClose()
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Kaydedilemedi')
    } finally { setSaving(false) }
  }

  const groups: { id: string; title: string; hint: string }[] = [
    { id: 'ner',   title: 'NER — Anlamsal Varlıklar', hint: 'Kişi, şirket, konum…' },
    { id: 'regex', title: 'Regex — Yapısal Varlıklar', hint: 'TC, IBAN, telefon, numaralar…' },
  ]
  const enabledCount  = data ? data.labels.filter(l => draft[l.label]).length : 0
  const disabledCount = data ? data.labels.length - enabledCount : 0
  const dirty = data ? data.labels.some(l => draft[l.label] !== data.policy[l.label]) : false

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 p-4"
      onMouseDown={onClose}
    >
      <div
        className="w-full max-w-2xl max-h-[90vh] overflow-hidden flex flex-col rounded-2xl border border-slate-700 bg-slate-900 shadow-2xl"
        onMouseDown={e => e.stopPropagation()}
      >
        {/* Header */}
        <div className="flex items-start justify-between px-5 py-4 border-b border-slate-800">
          <div>
            <h2 className="text-base font-semibold text-slate-100 flex items-center gap-2">
              ⚙️ Maskeleme Politikası
            </h2>
            <p className="text-xs text-slate-400 mt-1 leading-relaxed max-w-md">
              Hangi varlık türlerinin çıktıda maskeleneceğini seçin. Bu yalnızca
              gösterilen çıktıyı etkiler — model her şeyi tespit etmeye devam eder,
              eğitim verisi ve düzeltme kaydı değişmez.
            </p>
          </div>
          <button onClick={onClose}
            className="text-slate-400 hover:text-slate-100 transition-colors text-xl leading-none -mt-1">×</button>
        </div>

        {/* Toolbar */}
        <div className="flex items-center justify-between px-5 py-2.5 border-b border-slate-800 bg-slate-900/60">
          <span className="text-xs text-slate-400 tabular-nums">
            <span className="text-green-400">{enabledCount} maskeleniyor</span>
            {disabledCount > 0 && <> · <span className="text-amber-400">{disabledCount} kapalı</span></>}
          </span>
          <div className="flex items-center gap-2">
            <button onClick={() => setAll(true)}
              className="text-[11px] rounded-md border border-slate-700 bg-slate-800 px-2 py-1 text-slate-300 hover:bg-slate-700 transition-colors">
              Tümünü Aç
            </button>
            <button onClick={() => setAll(false)}
              className="text-[11px] rounded-md border border-slate-700 bg-slate-800 px-2 py-1 text-slate-300 hover:bg-slate-700 transition-colors">
              Tümünü Kapat
            </button>
          </div>
        </div>

        {/* Body */}
        <div className="flex-1 overflow-y-auto px-5 py-4 flex flex-col gap-5">
          {error && <p className="text-xs text-red-400">Hata: {error}</p>}
          {!data && !error && <p className="text-xs text-slate-400">Yükleniyor…</p>}

          {data && groups.map(group => {
            const items = data.labels.filter(l => l.group === group.id)
            if (items.length === 0) return null
            return (
              <div key={group.id}>
                <div className="flex items-baseline gap-2 mb-2">
                  <h3 className="text-xs font-semibold text-slate-300 uppercase tracking-wide">{group.title}</h3>
                  <span className="text-[10px] text-slate-500">{group.hint}</span>
                </div>
                <div className="grid grid-cols-2 gap-1.5">
                  {items.map(item => {
                    const on = draft[item.label] ?? true
                    const c  = labelColour(item.label)
                    return (
                      <button key={item.label}
                        onClick={() => toggle(item.label)}
                        className={`flex items-center justify-between rounded-lg border px-2.5 py-2 text-left transition-colors ${
                          on
                            ? 'bg-slate-800 border-slate-700 hover:border-slate-600'
                            : 'bg-slate-900 border-slate-800 hover:border-slate-700'
                        }`}>
                        <span className="flex items-center gap-2 min-w-0">
                          <span className={`inline-block w-2.5 h-2.5 rounded-full shrink-0 border ${c.bg} ${c.border}`} />
                          <span className={`text-xs truncate ${on ? 'text-slate-200' : 'text-slate-500'}`}>
                            {item.display}
                          </span>
                        </span>
                        {/* Toggle switch */}
                        <span className={`relative inline-flex h-4 w-7 shrink-0 items-center rounded-full transition-colors ${
                          on ? 'bg-blue-600' : 'bg-slate-700'
                        }`}>
                          <span className={`inline-block h-3 w-3 transform rounded-full bg-white transition-transform ${
                            on ? 'translate-x-3.5' : 'translate-x-0.5'
                          }`} />
                        </span>
                      </button>
                    )
                  })}
                </div>
              </div>
            )
          })}
        </div>

        {/* Footer actions */}
        <div className="flex items-center justify-between px-5 py-3 border-t border-slate-800">
          <p className="text-[10px] text-slate-500">
            Kaydedince yüklü belge yeni politikayla yeniden maskelenir.
          </p>
          <div className="flex items-center gap-2">
            <button onClick={onClose}
              className="rounded-lg border border-slate-700 px-3 py-2 text-xs text-slate-300 hover:bg-slate-800 transition-colors">
              İptal
            </button>
            <button onClick={save} disabled={saving || !dirty}
              className="rounded-lg bg-blue-600 px-4 py-2 text-xs font-semibold text-white hover:bg-blue-700 disabled:opacity-40 disabled:cursor-not-allowed transition-colors">
              {saving ? 'Kaydediliyor…' : 'Kaydet ve Uygula'}
            </button>
          </div>
        </div>
      </div>
    </div>
  )
}

// ── Chat sidebar ─────────────────────────────────────────────────────────────

interface ChatMsg { role: 'user' | 'assistant'; content: string }

interface ChatSidebarProps {
  maskedText: string
  onClose: () => void
}

function ChatSidebar({ maskedText, onClose }: ChatSidebarProps) {
  const [messages, setMessages] = useState<ChatMsg[]>([])
  const [input, setInput]       = useState('')
  const [busy, setBusy]         = useState(false)
  const bottomRef               = useRef<HTMLDivElement>(null)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, busy])

  async function send() {
    const text = input.trim()
    if (!text || busy) return
    const next: ChatMsg[] = [...messages, { role: 'user', content: text }]
    setMessages(next)
    setInput('')
    setBusy(true)
    try {
      const res = await fetch(`${API}/api/v1/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ masked_text: maskedText, messages: next }),
      })
      if (!res.ok) {
        const e = await res.json().catch(() => ({}))
        throw new Error(e.detail ?? `${res.status}`)
      }
      const data = await res.json()
      setMessages(m => [...m, { role: 'assistant', content: data.reply }])
    } catch (err) {
      setMessages(m => [...m, { role: 'assistant', content: `Hata: ${err instanceof Error ? err.message : 'Bilinmeyen hata'}` }])
    } finally {
      setBusy(false)
    }
  }

  function onKey(e: React.KeyboardEvent) {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); send() }
  }

  return (
    <div className="fixed top-0 right-0 h-full w-[360px] bg-slate-900 border-l border-slate-800 shadow-2xl flex flex-col z-40">
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-3 border-b border-slate-800 bg-slate-900">
        <div className="flex items-center gap-2">
          <span className="text-sm font-semibold text-slate-100">Belge Asistanı</span>
          <span className="text-[10px] text-slate-400 border border-slate-700 rounded px-1.5 py-0.5">maskelenmiş içerik</span>
        </div>
        <button onClick={onClose} className="text-slate-400 hover:text-slate-100 transition-colors text-lg leading-none">×</button>
      </div>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto p-4 flex flex-col gap-3">
        {messages.length === 0 && (
          <div className="flex flex-col items-center justify-center h-full gap-3 text-slate-500 text-center">
            <div className="w-10 h-10 rounded-xl bg-blue-950/60 flex items-center justify-center text-xl">💬</div>
            <p className="text-xs">Maskelenmiş belge hakkında soru sorabilirsiniz.<br/>Orijinal veriler asla OpenAI'ye gönderilmez.</p>
          </div>
        )}
        {messages.map((m, i) => (
          <div key={i} className={`flex ${m.role === 'user' ? 'justify-end' : 'justify-start'}`}>
            <div className={`max-w-[85%] rounded-2xl px-3.5 py-2.5 text-sm leading-relaxed whitespace-pre-wrap ${
              m.role === 'user'
                ? 'bg-blue-600 text-white rounded-br-sm'
                : 'bg-slate-800 text-slate-100 rounded-bl-sm'
            }`}>
              {m.content}
            </div>
          </div>
        ))}
        {busy && (
          <div className="flex justify-start">
            <div className="bg-slate-800 rounded-2xl rounded-bl-sm px-4 py-2.5 flex gap-1 items-center">
              {[0, 150, 300].map(d => (
                <span key={d} className="w-1.5 h-1.5 rounded-full bg-slate-500 animate-bounce" style={{ animationDelay: `${d}ms` }} />
              ))}
            </div>
          </div>
        )}
        <div ref={bottomRef} />
      </div>

      {/* Input */}
      <div className="px-3 py-3 border-t border-slate-800 flex gap-2 items-end">
        <textarea
          className="flex-1 resize-none rounded-xl border border-slate-700 bg-slate-800 px-3 py-2 text-sm text-slate-100 focus:outline-none focus:border-blue-500 transition-colors placeholder:text-slate-500 max-h-28"
          rows={1}
          placeholder="Soru sorun…"
          value={input}
          onChange={e => setInput(e.target.value)}
          onKeyDown={onKey}
          disabled={busy}
        />
        <button
          onClick={send}
          disabled={!input.trim() || busy}
          className="rounded-xl bg-blue-600 text-white px-3 py-2 text-sm font-medium hover:bg-blue-700 disabled:opacity-40 disabled:cursor-not-allowed transition-colors self-end"
        >
          ↑
        </button>
      </div>
    </div>
  )
}

// ── Main App ──────────────────────────────────────────────────────────────────

const SAMPLE_TEXT =
`Sayın Müşterimiz,
TC Kimlik No 10000000146 sahibi Ahmet Yılmaz'ın
TR330006100519786457841326 numaralı IBAN hesabına
15 Haziran 2024 tarihinde ₺12.500,00 transfer yapılmıştır.
İletişim: ahmet.yilmaz@banka.example.com veya 0532 123 45 67
Araç: 34 ABC 1234  Pasaport: U12345678
Şirket: Türk Hava Yolları A.Ş. — İstanbul ofisi`

type Tab       = 'masked' | 'spans' | 'unmask' | 'learning' | 'ablation'
type InputMode = 'text' | 'file'

export default function App() {
  const [inputMode,    setInputMode]    = useState<InputMode>('text')
  const [input,        setInput]        = useState('')
  const [result,       setResult]       = useState<MaskResponse | null>(null)
  const [uploadMeta,   setUploadMeta]   = useState<Pick<UploadMaskResponse, 'filename' | 'extraction' | 'has_ocr_pages'> | null>(null)
  const [loading,      setLoading]      = useState(false)
  const [error,        setError]        = useState<string | null>(null)
  const [tab,          setTab]          = useState<Tab>('masked')
  const [nerAvailable, setNerAvailable] = useState<boolean | null>(null)
  const [manualSpans,  setManualSpans]  = useState<ManualSpanRequest[]>([])
  const [popover,      setPopover]      = useState<PopoverState | null>(null)
  const [currentPage,  setCurrentPage]  = useState(0)
  const [chatOpen,     setChatOpen]     = useState(false)
  const [flaggedSpans, setFlaggedSpans] = useState<Set<string>>(new Set())
  const [adminOpen,    setAdminOpen]    = useState(false)
  // Global output policy: label → enabled. Detected-but-disabled entities are
  // shown unmasked. Fetched once on load, refreshed when the admin panel saves.
  const [maskPolicy,   setMaskPolicy]   = useState<Record<string, boolean>>({})

  const hasResult = result !== null

  // Labels the deployment has turned OFF — used to dim those spans in the
  // entities panel (they are detected but not masked in the output).
  const disabledLabels = useMemo(
    () => new Set(Object.entries(maskPolicy).filter(([, on]) => !on).map(([l]) => l)),
    [maskPolicy],
  )

  // Load the current mask policy once on mount.
  useEffect(() => {
    fetch(`${API}/api/v1/admin/mask-policy`)
      .then(r => r.json())
      .then(d => setMaskPolicy(d.policy ?? {}))
      .catch(() => {})
  }, [])

  // ── Correction API helper ─────────────────────────────────────────────────
  async function logCorrection(payload: {
    job_id?: string; event_type: string; entity_label: string
    entity_text: string; detector_source?: string | null
    corrected_label?: string | null; confidence?: number | null
    context?: string | null
  }) {
    try {
      await fetch(`${API}/api/v1/corrections`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      })
    } catch { /* fire-and-forget — never block the UI */ }
  }

  // Build a ±100-char window around a span — stored in training_examples so
  // corrections are usable for NER fine-tuning (the model learns from the
  // surrounding text, not the entity string alone).
  function extractContext(fullText: string, start: number, end: number): string {
    const CONTEXT_CHARS = 100
    const from = Math.max(0, start - CONTEXT_CHARS)
    const to   = Math.min(fullText.length, end + CONTEXT_CHARS)
    const prefix = from > 0 ? '…' : ''
    const suffix = to < fullText.length ? '…' : ''
    return prefix + fullText.slice(from, to).trim() + suffix
  }

  // Recompute page count from result whenever it changes
  const pages = useMemo(
    () => result ? splitIntoPages(result.original_text, result.page_separator ?? null) : [],
    [result]
  )
  const totalPages = pages.length || 1

  // ── API call ──────────────────────────────────────────────────────────────
  async function doMask(text: string, manual: ManualSpanRequest[]) {
    setLoading(true); setError(null)
    try {
      const res = await fetch(`${API}/api/v1/mask`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text, language: 'tr', mode: 'full', manual_spans: manual }),
      })
      if (!res.ok) { const e = await res.json().catch(() => ({})); throw new Error(e.detail ?? `${res.status}`) }
      const data: MaskResponse = await res.json()
      setResult(data); setTab('masked'); setCurrentPage(0)
      fetch(`${API}/health`)
        .then(r => r.json())
        .then(h => setNerAvailable(h.ner_available ?? false))
        .catch(() => {})
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unknown error')
    } finally { setLoading(false) }
  }

  // ── File upload ───────────────────────────────────────────────────────────
  async function doUpload(file: File) {
    setLoading(true); setError(null); setUploadMeta(null)
    try {
      const form = new FormData()
      form.append('file', file)
      form.append('mode', 'full')
      const res = await fetch(`${API}/api/v1/upload`, { method: 'POST', body: form })
      if (!res.ok) { const e = await res.json().catch(() => ({})); throw new Error(e.detail ?? `${res.status}`) }
      const data: UploadMaskResponse = await res.json()
      setInput(data.original_text)
      setResult(data)
      setUploadMeta({ filename: data.filename, extraction: data.extraction, has_ocr_pages: data.has_ocr_pages })
      setTab('masked'); setCurrentPage(0)
      setNerAvailable(true)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unknown error')
    } finally { setLoading(false) }
  }

  function handleMask() { if (input.trim()) doMask(input, manualSpans) }

  // ── Flag an auto-detected span as false positive ──────────────────────────
  function handleFlag(span: Span) {
    const key = `${span.start}:${span.end}:${span.label}`
    setFlaggedSpans(prev => new Set([...prev, key]))
    logCorrection({
      job_id:          result?.job_id,
      event_type:      'false_positive',
      entity_label:    span.label,
      entity_text:     span.text,
      detector_source: span.source,
      confidence:      span.confidence,
      context:         result ? extractContext(result.original_text, span.start, span.end) : null,
    })
  }

  // ── Manual span confirmed in popover ─────────────────────────────────────
  const handleManualConfirm = useCallback((text: string, label: string) => {
    setPopover(null)
    const updated = [...manualSpans, { text, label }]
    setManualSpans(updated)
    doMask(input, updated)
    // Log as false negative — user found something the detectors missed.
    // Manual spans are text-based (masked at every occurrence); use the first
    // occurrence in the document for the training context window.
    const idx = input.indexOf(text)
    logCorrection({
      job_id:          result?.job_id,
      event_type:      'false_negative',
      entity_label:    label,
      entity_text:     text,
      detector_source: null,
      context:         idx >= 0 ? extractContext(input, idx, idx + text.length) : null,
    })
  }, [manualSpans, input, result])

  // ── Remove a manual span ──────────────────────────────────────────────────
  function removeManualSpan(spanText: string, label: string) {
    const updated = manualSpans.filter(s => !(s.text === spanText && s.label === label))
    setManualSpans(updated)
    doMask(input, updated)
  }

  function handleClear() {
    setInput(''); setResult(null); setError(null); setUploadMeta(null)
    setManualSpans([]); setPopover(null); setCurrentPage(0); setChatOpen(false)
    setFlaggedSpans(new Set())
  }

  const regexCount  = result?.spans.filter(s => s.source === 'regex').length  ?? 0
  const nerCount    = result?.spans.filter(s => s.source === 'ner').length    ?? 0
  const manualCount = result?.spans.filter(s => s.source === 'manual').length ?? 0
  const gazCount    = result?.spans.filter(s => s.source === 'gazetteer').length ?? 0

  return (
    // No onClick handler here — popover handles its own close logic.
    // h-screen (not min-h-screen): the body is viewport-locked so each panel
    // scrolls internally — the left action bar never gets pushed off-screen
    // by tall right-panel content (e.g. a long entity list).
    <div className="h-screen bg-slate-950 flex flex-col">

      {/* ── Header ── */}
      <header className="bg-slate-900 border-b border-slate-800 px-6 py-3 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="w-7 h-7 rounded bg-blue-600 flex items-center justify-center">
            <span className="text-white text-xs font-bold">ML</span>
          </div>
          <span className="font-semibold text-slate-100 text-lg">MaskLayer</span>
          <span className="text-xs text-slate-400 border border-slate-700 rounded px-1.5 py-0.5">
            Turkish PII Masking
          </span>
        </div>
        <div className="flex items-center gap-3 text-xs">
          {nerAvailable !== null && (
            <span className={`flex items-center gap-1 rounded-full px-2 py-0.5 border ${
              nerAvailable
                ? 'bg-green-50 text-green-700 border-green-200'
                : 'bg-yellow-50 text-yellow-700 border-yellow-200'
            }`}>
              <span className={`w-1.5 h-1.5 rounded-full ${nerAvailable ? 'bg-green-500' : 'bg-yellow-500'}`}/>
              {nerAvailable ? 'NER aktif' : 'Regex only'}
            </span>
          )}
          {result && (
            <span className="text-slate-500 tabular-nums">{result.processing_time_ms.toFixed(0)} ms</span>
          )}
          <button
            onClick={() => setAdminOpen(true)}
            title="Maskeleme politikası — hangi türlerin maskeleneceğini ayarlayın"
            className="flex items-center gap-1.5 rounded-lg border px-3 py-1.5 text-xs font-medium transition-colors bg-slate-900 text-slate-300 border-slate-700 hover:bg-slate-800">
            ⚙️ Politika
            {disabledLabels.size > 0 && (
              <span className="rounded-full bg-amber-500/20 text-amber-300 border border-amber-500/40 px-1.5 py-px text-[10px] tabular-nums">
                {disabledLabels.size} kapalı
              </span>
            )}
          </button>
          <button
            onClick={() => setChatOpen(o => !o)}
            disabled={!hasResult}
            title={hasResult ? 'Belge asistanını aç' : 'Önce bir belge maskeleyin'}
            className={`flex items-center gap-1.5 rounded-lg border px-3 py-1.5 text-xs font-medium transition-colors disabled:opacity-40 disabled:cursor-not-allowed ${
              chatOpen
                ? 'bg-blue-600 text-white border-blue-600'
                : 'bg-slate-900 text-slate-300 border-slate-700 hover:bg-slate-800'
            }`}>
            💬 Asistan
          </button>
        </div>
      </header>

      {/* ── Body ── */}
      <div className="flex flex-1 overflow-hidden">

        {/* ── Left panel ── */}
        <div className="flex flex-col w-[44%] border-r border-slate-800 bg-slate-900">

          {/* Mode tabs + contextual actions */}
          <div className="px-4 py-2 border-b border-slate-800 flex items-center justify-between min-h-[40px]">
            {/* Text / File toggle */}
            <div className="flex rounded-lg border border-slate-700 overflow-hidden text-xs">
              {(['text', 'file'] as InputMode[]).map(m => (
                <button key={m}
                  onClick={() => { if (!hasResult) setInputMode(m) }}
                  disabled={hasResult}
                  className={`px-3 py-1.5 font-medium transition-colors disabled:opacity-50 disabled:cursor-not-allowed ${
                    inputMode === m
                      ? 'bg-blue-600 text-white'
                      : 'bg-slate-900 text-slate-400 hover:bg-slate-800'
                  }`}>
                  {m === 'text' ? '✏️ Metin' : '📂 Dosya'}
                </button>
              ))}
            </div>

            <div className="flex items-center gap-2">
              {hasResult && (
                <span className="text-[10px] bg-amber-50 text-amber-700 border border-amber-200 rounded-full px-2 py-0.5">
                  Metni seçip maskele
                </span>
              )}
              {!hasResult && inputMode === 'text' && (
                <button onClick={() => setInput(SAMPLE_TEXT)}
                  className="text-xs text-blue-400 hover:text-blue-300 transition-colors">
                  Örnek metni kullan
                </button>
              )}
            </div>
          </div>

          {/* Content area */}
          {hasResult ? (
            <AnnotatedText
              text={input}
              spans={result.spans}
              pageSep={result.page_separator ?? null}
              extraction={uploadMeta?.extraction}
              currentPage={currentPage}
              onAddSpan={(selected, x, y, yAbove) => setPopover({ selectedText: selected, x, y, yAbove })}
            />
          ) : inputMode === 'file' ? (
            <DropZone onFile={doUpload} loading={loading} />
          ) : (
            <textarea
              className="flex-1 resize-none p-4 text-sm text-slate-100 bg-transparent font-mono focus:outline-none placeholder:text-slate-600"
              placeholder="Metni buraya yapıştırın…"
              value={input}
              onChange={e => setInput(e.target.value)}
            />
          )}

          {/* Action bar — only shown in text mode or post-result */}
          {(inputMode === 'text' || hasResult) && (
            <div className="px-4 py-3 border-t border-slate-800 flex items-center gap-3">
              <button onClick={handleMask} disabled={loading || !input.trim()}
                className="flex-1 rounded-lg bg-blue-600 py-2.5 text-sm font-semibold text-white hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors">
                {loading ? (
                  <span className="flex items-center justify-center gap-2">
                    <svg className="animate-spin h-4 w-4" viewBox="0 0 24 24" fill="none">
                      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"/>
                      <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8H4z"/>
                    </svg>
                    Maskeleniyor…
                  </span>
                ) : hasResult ? 'Yeniden Maskele' : 'Maskele'}
              </button>
              {hasResult && (
                <button onClick={() => { setResult(null); setUploadMeta(null); setManualSpans([]) }}
                  className="text-xs text-slate-400 hover:text-slate-100 border border-slate-700 rounded-lg px-3 py-2.5 bg-slate-900 hover:bg-slate-800 transition-colors">
                  Düzenle
                </button>
              )}
              {input && (
                <button onClick={handleClear}
                  className="text-xs text-slate-400 hover:text-red-400 transition-colors">
                  Temizle
                </button>
              )}

              {/* Page navigation — only when multi-page */}
              {hasResult && totalPages > 1 && (
                <div className="flex items-center gap-1 ml-auto">
                  <button
                    onClick={() => setCurrentPage(p => Math.max(0, p - 1))}
                    disabled={currentPage === 0}
                    className="w-8 h-8 flex items-center justify-center rounded-lg border border-slate-700 bg-slate-900 text-slate-300 hover:bg-slate-800 disabled:opacity-30 disabled:cursor-not-allowed transition-colors text-base leading-none"
                    title="Önceki sayfa">
                    ‹
                  </button>
                  <span className="text-xs tabular-nums text-slate-500 w-12 text-center">
                    {currentPage + 1} / {totalPages}
                  </span>
                  <button
                    onClick={() => setCurrentPage(p => Math.min(totalPages - 1, p + 1))}
                    disabled={currentPage === totalPages - 1}
                    className="w-8 h-8 flex items-center justify-center rounded-lg border border-slate-700 bg-slate-900 text-slate-300 hover:bg-slate-800 disabled:opacity-30 disabled:cursor-not-allowed transition-colors text-base leading-none"
                    title="Sonraki sayfa">
                    ›
                  </button>
                </div>
              )}
            </div>
          )}

          {error && (
            <div className="mx-4 mb-3 rounded-lg bg-red-950/60 border border-red-900 px-3 py-2 text-xs text-red-300">
              {error}
            </div>
          )}
        </div>

        {/* ── Right panel ── */}
        <div className="flex flex-col flex-1 bg-slate-950">

          {/* Tab bar */}
          <div className="flex items-center border-b border-slate-800 bg-slate-900 px-4">
            {([
              ['masked',   'Maskelenmiş Metin'],
              ['spans',    `Varlıklar (${result?.spans.length ?? 0})`],
              ['unmask',   'Maskeyi Kaldır'],
              ['ablation', '📊 Ablasyon'],
              ['learning', '🎓 Öğrenme'],
            ] as [Tab, string][]).map(([id, label]) => (
              <button key={id} onClick={() => setTab(id)}
                disabled={!result && id !== 'masked' && id !== 'learning' && id !== 'ablation'}
                className={`px-4 py-3 text-xs font-semibold border-b-2 transition-colors ${
                  tab === id
                    ? 'border-blue-500 text-blue-400'
                    : 'border-transparent text-slate-500 hover:text-slate-300 disabled:opacity-30 disabled:cursor-not-allowed'
                }`}>
                {label}
              </button>
            ))}
            {result && (
              <div className="ml-auto flex items-center gap-2 py-2">
                {regexCount > 0  && <span className="rounded-full bg-blue-100   text-blue-700   text-[10px] font-medium px-2 py-0.5 border border-blue-200">{regexCount} regex</span>}
                {nerCount > 0    && <span className="rounded-full bg-purple-100 text-purple-700 text-[10px] font-medium px-2 py-0.5 border border-purple-200">{nerCount} NER</span>}
                {gazCount > 0    && <span className="rounded-full bg-teal-100   text-teal-700   text-[10px] font-medium px-2 py-0.5 border border-teal-200">{gazCount} sözlük</span>}
                {manualCount > 0 && <span className="rounded-full bg-orange-100 text-orange-700 text-[10px] font-medium px-2 py-0.5 border border-orange-200">{manualCount} elle</span>}
              </div>
            )}
          </div>

          {/* Tab content */}
          <div className="flex-1 overflow-auto p-6">
            {!result && !loading && (
              <div className="flex flex-col items-center justify-center h-full gap-4 text-slate-500">
                <div className="w-12 h-12 rounded-xl bg-slate-800 flex items-center justify-center text-2xl">🔒</div>
                <div className="text-center">
                  <p className="text-sm font-medium">Henüz metin maskelenmedi</p>
                  <p className="text-xs mt-1">Metin girin ve "Maskele" butonuna basın</p>
                  <p className="text-xs mt-0.5 text-amber-600">Maskeledikten sonra sol panelden metin seçerek elle maske ekleyebilirsiniz</p>
                </div>
                <Legend />
              </div>
            )}

            {loading && (
              <div className="flex flex-col items-center justify-center h-full gap-3 text-slate-400">
                <svg className="animate-spin h-8 w-8 text-blue-500" viewBox="0 0 24 24" fill="none">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"/>
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8H4z"/>
                </svg>
                <p className="text-sm">Regex + NER çalışıyor…</p>
              </div>
            )}

            {result && tab === 'masked' && (
              <div className="flex flex-col gap-4">
                {uploadMeta && (
                  <div className="flex items-center gap-2 text-xs text-blue-400">
                    <span>📄</span>
                    <span className="font-medium truncate max-w-xs" title={uploadMeta.filename}>{uploadMeta.filename}</span>
                    <span className="text-blue-400">·</span>
                    <span className="text-blue-500">{uploadMeta.extraction.length} sayfa</span>
                    {uploadMeta.has_ocr_pages && (
                      <span className="rounded-full bg-amber-100 text-amber-700 border border-amber-200 px-2 py-0.5">OCR içeriyor</span>
                    )}
                  </div>
                )}
                <MaskedTextView
                  text={result.masked_text}
                  spans={result.spans}
                  pageSep={result.page_separator ?? null}
                  extraction={uploadMeta?.extraction}
                  currentPage={currentPage}
                />
                <div className="flex justify-end">
                  <button onClick={() => navigator.clipboard.writeText(result.masked_text)}
                    className="text-xs text-slate-300 border border-slate-700 rounded px-3 py-1.5 bg-slate-800 hover:bg-slate-700 transition-colors">
                    Kopyala
                  </button>
                </div>
                <div className="rounded-xl border border-slate-700 bg-slate-800 p-4 shadow-sm">
                  <p className="text-xs font-semibold text-gray-400 uppercase tracking-wide mb-2">Renk Açıklaması</p>
                  <Legend />
                </div>
              </div>
            )}

            {result && tab === 'spans' && (
              <div className="flex flex-col gap-3">
                <p className="text-xs text-slate-400">
                  {result.spans.length} varlık —{' '}
                  <span className="text-blue-400">{regexCount} regex</span>
                  {nerCount    > 0 && <> · <span className="text-purple-400">{nerCount} NER</span></>}
                  {gazCount    > 0 && <> · <span className="text-teal-400">{gazCount} sözlük</span></>}
                  {manualCount > 0 && <> · <span className="text-orange-400">{manualCount} elle</span></>}
                </p>
                <SpanList
                  spans={result.spans}
                  onRemove={s => removeManualSpan(s.text, s.label)}
                  onFlag={handleFlag}
                  flagged={flaggedSpans}
                  disabledLabels={disabledLabels}
                />
              </div>
            )}

            {result && tab === 'unmask' && <UnmaskPanel jobId={result.job_id} />}

            {tab === 'ablation' && <AblationPanel text={input} key={tab} />}

            {tab === 'learning' && <LearningPanel key={tab} />}
          </div>
        </div>
      </div>

      {/* ── Footer ── */}
      <footer className="bg-slate-900 border-t border-slate-800 px-6 py-2 flex items-center justify-between text-xs text-slate-500">
        <span>MaskLayer — Türkçe KVKK & BDDK uyumlu PII maskeleme</span>
        {result && <span className="font-mono">job: {result.job_id.slice(0, 8)}…</span>}
      </footer>

      {/* ── Chat sidebar ── */}
      {chatOpen && result && (
        <ChatSidebar maskedText={result.masked_text} onClose={() => setChatOpen(false)} />
      )}

      {/* ── Admin / mask-policy panel ── */}
      {adminOpen && (
        <AdminPanel
          onClose={() => setAdminOpen(false)}
          onSaved={(policy) => {
            setMaskPolicy(policy)
            // Re-mask the loaded document so the new policy is reflected
            // immediately. input holds the original text in both text and
            // upload modes; detection re-runs, only the output filter differs.
            if (hasResult && input.trim()) doMask(input, manualSpans)
          }}
        />
      )}

      {/* ── Floating popover (rendered at root, outside all other handlers) ── */}
      {popover && (
        <LabelPopover
          popover={popover}
          onConfirm={handleManualConfirm}
          onClose={() => setPopover(null)}
        />
      )}
    </div>
  )
}
