export interface Span {
  start: number
  end: number
  label: string
  source: 'regex' | 'ner' | 'manual' | 'gazetteer'
  confidence: number
  text: string
  canonical_id: string | null
}

export interface ManualSpanRequest {
  text: string
  label: string
}

export interface MaskRequest {
  text: string
  language: string
  mode: 'full' | 'regex' | 'ner'
  manual_spans: ManualSpanRequest[]
}

export interface MaskResponse {
  job_id: string
  original_text: string
  masked_text: string
  spans: Span[]
  processing_time_ms: number
  // Page layout — null for plain text input, populated by upload endpoint
  page_breaks: number[] | null
  page_separator: string | null
}

export interface UnmaskResponse {
  job_id: string
  unmasked_text: string
}

export interface PageExtractionInfo {
  page_number: number
  method: 'native' | 'ocr' | 'docx'
  confidence: number | null   // Tesseract mean confidence (0–100), null for native/docx
}

// Response from POST /api/v1/upload — superset of MaskResponse
export interface UploadMaskResponse extends MaskResponse {
  filename: string
  extraction: PageExtractionInfo[]
  has_ocr_pages: boolean
  page_breaks: number[]   // always present (override nullable from MaskResponse)
  page_separator: string  // always "\f"
}

// ── Entity taxonomy ───────────────────────────────────────────────────────────

export const TAXONOMY_LABELS = [
  { label: 'Person',         display: 'Kişi',          group: 'ner'   },
  { label: 'Company',        display: 'Şirket',         group: 'ner'   },
  { label: 'Location',       display: 'Konum',          group: 'ner'   },
  { label: 'Title',          display: 'Ünvan',          group: 'ner'   },
  { label: 'Facility',       display: 'Tesis',          group: 'ner'   },
  { label: 'TC_No',          display: 'TC Kimlik',      group: 'regex' },
  { label: 'IBAN',           display: 'IBAN',           group: 'regex' },
  { label: 'Tax_No',         display: 'Vergi No',       group: 'regex' },
  { label: 'Card_No',        display: 'Kart No',        group: 'regex' },
  { label: 'Phone_No',       display: 'Telefon',        group: 'regex' },
  { label: 'Email',          display: 'E-posta',        group: 'regex' },
  { label: 'Date',           display: 'Tarih',          group: 'regex' },
  { label: 'Money_Amount',   display: 'Para',           group: 'regex' },
  { label: 'Account_No',     display: 'Hesap No',       group: 'regex' },
  { label: 'License_Plate',  display: 'Plaka',          group: 'regex' },
  { label: 'Passport_No',    display: 'Pasaport',       group: 'regex' },
  { label: 'Case_No',        display: 'Dava No',        group: 'regex' },
  { label: 'Address',        display: 'Adres',           group: 'regex' },
  { label: 'Customer_No',    display: 'Müşteri No',     group: 'regex' },
  { label: 'Contract_No',    display: 'Sözleşme No',    group: 'regex' },
  { label: 'Policy_No',      display: 'Poliçe No',      group: 'regex' },
  { label: 'Invoice_No',     display: 'Fatura No',      group: 'regex' },
  { label: 'SGK_No',         display: 'SGK No',         group: 'regex' },
  { label: 'Reference_No',   display: 'Referans No',    group: 'regex' },
]

// ── Colour palette ────────────────────────────────────────────────────────────

export const LABEL_COLOURS: Record<string, { bg: string; text: string; border: string }> = {
  TC_No:          { bg: 'bg-red-100',     text: 'text-red-800',     border: 'border-red-300'     },
  IBAN:           { bg: 'bg-orange-100',  text: 'text-orange-800',  border: 'border-orange-300'  },
  Card_No:        { bg: 'bg-amber-100',   text: 'text-amber-800',   border: 'border-amber-300'   },
  Tax_No:         { bg: 'bg-yellow-100',  text: 'text-yellow-800',  border: 'border-yellow-300'  },
  Phone_No:       { bg: 'bg-lime-100',    text: 'text-lime-800',    border: 'border-lime-300'    },
  Email:          { bg: 'bg-green-100',   text: 'text-green-800',   border: 'border-green-300'   },
  License_Plate:  { bg: 'bg-teal-100',    text: 'text-teal-800',    border: 'border-teal-300'    },
  Date:           { bg: 'bg-cyan-100',    text: 'text-cyan-800',    border: 'border-cyan-300'    },
  Money_Amount:   { bg: 'bg-sky-100',     text: 'text-sky-800',     border: 'border-sky-300'     },
  Case_No:        { bg: 'bg-blue-100',    text: 'text-blue-800',    border: 'border-blue-300'    },
  Account_No:     { bg: 'bg-indigo-100',  text: 'text-indigo-800',  border: 'border-indigo-300'  },
  Passport_No:    { bg: 'bg-violet-100',  text: 'text-violet-800',  border: 'border-violet-300'  },
  Person:         { bg: 'bg-purple-100',  text: 'text-purple-800',  border: 'border-purple-300'  },
  Company:        { bg: 'bg-fuchsia-100', text: 'text-fuchsia-800', border: 'border-fuchsia-300' },
  Location:       { bg: 'bg-pink-100',    text: 'text-pink-800',    border: 'border-pink-300'    },
  Title:          { bg: 'bg-rose-100',    text: 'text-rose-800',    border: 'border-rose-300'    },
  Facility:       { bg: 'bg-slate-100',   text: 'text-slate-800',   border: 'border-slate-300'   },
  IP_Address:     { bg: 'bg-gray-100',    text: 'text-gray-800',    border: 'border-gray-300'    },
  URL:            { bg: 'bg-zinc-100',    text: 'text-zinc-800',    border: 'border-zinc-300'    },
  Address:        { bg: 'bg-orange-50',   text: 'text-orange-900',  border: 'border-orange-200'  },
  Customer_No:    { bg: 'bg-emerald-100', text: 'text-emerald-800', border: 'border-emerald-300' },
  Contract_No:    { bg: 'bg-stone-100',   text: 'text-stone-800',   border: 'border-stone-300'   },
  Policy_No:      { bg: 'bg-sky-50',      text: 'text-sky-900',     border: 'border-sky-200'     },
  Invoice_No:     { bg: 'bg-amber-50',    text: 'text-amber-900',   border: 'border-amber-200'   },
  SGK_No:         { bg: 'bg-lime-50',     text: 'text-lime-900',    border: 'border-lime-200'    },
  Reference_No:   { bg: 'bg-cyan-50',     text: 'text-cyan-900',    border: 'border-cyan-200'    },
}

export function labelColour(label: string) {
  return LABEL_COLOURS[label] ?? { bg: 'bg-gray-100', text: 'text-gray-700', border: 'border-gray-300' }
}
