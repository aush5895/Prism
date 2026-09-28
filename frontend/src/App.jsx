import { useEffect, useMemo, useState } from 'react'
import {
  API_BASE, fetchSamples, guidedAnswer, guidedConfirm, guidedEscalate, guidedStart,
  troubleshoot,
} from './api.js'
import {
  EnrichmentPanel, GroundingPanel, PlanPanel, ResolverPanel, TelemetryPanel,
} from './panels.jsx'
import { AnalyzingCard, CustomerView, EntryCard } from './customer.jsx'
import { GuidedView } from './guided.jsx'

const REPO_URL = 'https://github.com/aush5895/Prism'

// The six demo beats, in the order the demo walks them. They are STATES OF THIS UI, not
// stages of the pipeline -- the pipeline's own stages are in the engineer view's
// telemetry panel, where they are measured. Clickable so the demo can jump to a beat.
const BEATS = [
  { n: 1, name: 'Complaint Entry', hint: 'Type or pick a supplied complaint' },
  { n: 2, name: 'Analyzing', hint: 'Run the pipeline on this complaint' },
  { n: 3, name: 'Results', hint: 'The ordered plan' },
  { n: 4, name: 'Step Detail', hint: 'Focus the first action' },
  { n: 5, name: 'Resolution Verify', hint: 'Show the catalog entry behind a link' },
  // Beat 6 used to be "Honest Fallback" on row_1, framed as the engine declining a
  // mismatched article. With the live provider row_1 returns a five-action plan, so the
  // beat demonstrated something the system does not do. It is gone; see LIMITATIONS.md.
  { n: 6, name: 'Guided Session', hint: 'Walk the plan one step at a time' },
  { n: 7, name: 'Agent Handoff', hint: 'Hand the session to a person with its full record' },
]

function ViewToggle({ view, setView }) {
  return (
    <div className="viewtoggle">
      <button className={view === 'customer' ? 'on' : ''} onClick={() => setView('customer')}>
        Customer view
      </button>
      <button className={view === 'engineer' ? 'on' : ''} onClick={() => setView('engineer')}>
        Engineer view
      </button>
    </div>
  )
}

export default function App() {
  const [samples, setSamples] = useState([])
  const [selected, setSelected] = useState('')
  const [query, setQuery] = useState('')
  const [result, setResult] = useState(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState(null)
  const [activeStep, setActiveStep] = useState(null)
  // Kept so the telemetry panel can state the cache speed-up as a measured ratio
  // rather than a claim. Reset whenever the complaint changes.
  const [lastColdMs, setLastColdMs] = useState(null)
  // Customer view is the default: the product is the plan, not the instrumentation.
  const [view, setView] = useState('customer')
  const [beat, setBeat] = useState(1)
  // Guided mode: the session the backend owns, and the envelope + catalog ids it walks.
  const [guided, setGuided] = useState(null)
  const [guidedBusy, setGuidedBusy] = useState(false)

  useEffect(() => {
    fetchSamples()
      .then((data) => {
        setSamples(data.samples)
        // row_21 is the strongest first impression: a full ten-action plan with real
        // deeplinks, and the kit's richest set of critical steps for guided mode.
        const first = data.samples.find((s) => s.id === 'row_21') || data.samples[0]
        if (first) {
          setSelected(first.id)
          setQuery(first.query)
        }
      })
      .catch((e) => setError(`Cannot reach the API at ${API_BASE}. Is it running? (${e.message})`))
  }, [])

  // The customer view is light, the engineer view dark, and the page background lives on
  // <body>, outside React's tree. This is the only place that reaches for it.
  useEffect(() => { document.body.dataset.view = view }, [view])

  const sample = samples.find((s) => s.id === selected)

  function pickSample(id) {
    const next = samples.find((s) => s.id === id)
    setSelected(id)
    if (next) setQuery(next.query)
    setResult(null)
    setGuided(null)
    setLastColdMs(null)
    setBeat(1)
  }

  // Every guided call returns the whole session view; the UI just renders the latest.
  async function guidedCall(fn) {
    if (guidedBusy) return
    setGuidedBusy(true)
    setError(null)
    try {
      const next = await fn()
      const session = next.session || next
      setGuided((prev) => ({ ...prev, session }))
      // A session that ends in a handoff lands on the handoff beat, however it got there.
      if (['escalated', 'no_plan'].includes(session.status)) setBeat(7)
      return next
    } catch (e) {
      setError(e.message)
      return undefined
    } finally {
      setGuidedBusy(false)
    }
  }

  async function startGuided(opts = {}) {
    const target = opts.sample || sample
    const text = opts.query ?? query
    if (!target) return undefined
    setGuidedBusy(true)
    setError(null)
    try {
      // The backend runs the normal pipeline first, so a complaint just analyzed is
      // answered from cache and the walk opens on the plan the customer already saw.
      const data = await guidedStart(text, target.siis_response)
      setGuided({ session: data.session, envelope: data.envelope, catalogIds: data.catalog_ids })
      setBeat(data.session.status === 'no_plan' ? 7 : 6)
      return data.session
    } catch (e) {
      setError(e.message)
      return undefined
    } finally {
      setGuidedBusy(false)
    }
  }

  function restart() {
    setGuided(null)
    setBeat(1)
  }

  async function analyze(opts = {}) {
    const target = opts.sample || sample
    const text = opts.query ?? query
    if (!target || busy) return
    setBusy(true)
    setBeat(2)
    setError(null)
    try {
      const data = await troubleshoot(text, target.siis_response)
      if (!data.envelope.meta.cache_hit) setLastColdMs(data.envelope.meta.latency_ms)
      setResult(data)
      const hasPlan = Boolean(data.envelope.response?.contexts?.[0])
      setBeat(opts.land ?? (hasPlan ? 3 : 6))
    } catch (e) {
      setError(e.message)
      setBeat(1)
    } finally {
      setBusy(false)
    }
  }

  function goBeat(n) {
    if (busy) return undefined
    if (n === 1) return setBeat(1)
    if (n === 2) return analyze()
    if (n === 6) {
      if (guided && guided.session.status !== 'resolved') return setBeat(6)
      return startGuided()
    }
    if (n === 7) {
      // Hand the current session to an agent; open one first if none is running.
      return (async () => {
        let session = guided?.session
        if (!session || ['resolved', 'escalated', 'no_plan'].includes(session.status)) {
          session = await startGuided()
        }
        if (session && !['escalated', 'no_plan'].includes(session.status)) {
          await guidedCall(() => guidedEscalate(session.session_id))
        }
        setBeat(7)
      })()
    }
    if (!result) return analyze({ land: n })
    return setBeat(n)
  }

  // Flatten the plan's steps once, pairing each with the span the backend verified for
  // it, so the article and the step list can highlight each other by a shared key.
  const steps = useMemo(() => {
    if (!result) return []
    const spans = result.debug.spans || []
    const actions = result.envelope.response?.contexts?.[0]?.actions || []
    const out = []
    actions.forEach((action, ai) => {
      action.stepGroups.forEach((group, gi) => {
        group.steps.forEach((text, si) => {
          const span = spans.find(
            (s) => s.action_index === ai && s.group_index === gi && s.step_index === si,
          )
          out.push({
            key: `${ai}.${gi}.${si}`,
            text,
            action: action.actionName,
            span: span && span.start !== null ? span : null,
          })
        })
      })
    })
    return out
  }, [result])

  // Catalog ids are not in the graded response (schema.py has no field for them), so the
  // proof panel reads them from the debug sibling, keyed by URI. A real client would not
  // have this; the demo does, and demonstrating it is the panel's whole job.
  const catalogIds = result?.debug?.catalog_ids || {}

  const markedSpans = useMemo(
    () => steps.filter((s) => s.span).map((s) => ({ ...s.span, key: s.key })),
    [steps],
  )

  // The model on the entry card is enrich.py's parse, never a guess made here: either the
  // slot the backend returned for the text we just ran, or the one /v1/samples carried
  // for a supplied complaint. Anything else is reported as not parsed yet.
  const detected = useMemo(() => {
    const text = query.trim()
    if (!text) return { parsed: false, device: null }
    if (result && (result.envelope.query || '').trim() === text) {
      return { parsed: true, device: result.debug?.enrichment?.device || null }
    }
    const match = samples.find((s) => s.query.trim() === text)
    if (match) return { parsed: true, device: match.device || null }
    return { parsed: false, device: null }
  }, [query, samples, result])

  const customerBeat = busy ? 2 : beat
  const showGuided = customerBeat >= 6 && Boolean(guided)
  const showResults = customerBeat >= 3 && Boolean(result) && !showGuided

  return (
    <div className={`app app--${view}`}>
      {view === 'customer' ? (
        <>
          <header className="pbar">
            <div className="lockup">
              <div className="lockup-name">SMART GUIDED TROUBLESHOOTING</div>
              <div className="lockup-sub">
                Samsung PRISM GenAI Hackathon · Theme 02 · grounded troubleshooting plans
              </div>
            </div>
            <span className="spacer" />
            <span
              className="devchip"
              title="No device is connected. The plan is produced from the complaint text and the service record supplied with the request."
            >
              Galaxy S22 · simulated
            </span>
            <ViewToggle view={view} setView={setView} />
          </header>

          <nav className="beats" aria-label="Engine simulation state">
            <span className="beats-label">Engine simulation state</span>
            <ol>
              {BEATS.map((b) => (
                <li key={b.n}>
                  <button
                    type="button"
                    className={`beat${customerBeat === b.n ? ' on' : ''}`}
                    title={b.hint}
                    disabled={busy || guidedBusy || !samples.length}
                    aria-current={customerBeat === b.n ? 'step' : undefined}
                    onClick={() => goBeat(b.n)}
                  >
                    <span className="beat-n">{b.n}</span>
                    <span className="beat-name">{b.name}</span>
                  </button>
                </li>
              ))}
            </ol>
          </nav>
        </>
      ) : (
        <header className="top">
          <h1>Smart Guided Troubleshooting Engine</h1>
          <span className="sub">Samsung PRISM GenAI Hackathon · Theme 02</span>
          <span className="spacer" />
          <ViewToggle view={view} setView={setView} />
          <span className="sub mono">{API_BASE}</span>
        </header>
      )}

      {view === 'engineer' && (
        <>
          <div className="controls">
            <select value={selected} onChange={(e) => pickSample(e.target.value)}>
              {samples.map((s) => (
                <option key={s.id} value={s.id}>{s.id} — {s.query.slice(0, 68)}…</option>
              ))}
            </select>
            <input
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="Reword the complaint to exercise the semantic cache…"
            />
            <button onClick={() => analyze()} disabled={busy || !sample}>
              {busy ? 'Running…' : 'Run'}
            </button>
            <button
              className="ghost"
              onClick={() => analyze()}
              disabled={busy || !sample || !result}
            >
              Run again (cache)
            </button>
          </div>
          <p className="hint">
            Run once to see the cold path, then reword the complaint and run again: the
            same article with a different phrasing should be answered from cache.
          </p>
        </>
      )}

      {error && <div className="err">{error}</div>}

      {view === 'customer' && (
        <main className="stage">
          {busy && <AnalyzingCard query={query} />}

          {!busy && !showResults && !showGuided && (
            <EntryCard
              query={query}
              setQuery={setQuery}
              samples={samples}
              selected={selected}
              onPick={pickSample}
              onSubmit={() => analyze()}
              detected={detected}
              disabled={!sample}
            />
          )}

          {!busy && showResults && (
            <CustomerView
              envelope={result.envelope}
              catalogIds={catalogIds}
              focus={customerBeat === 4 ? 'detail' : customerBeat === 5 ? 'verify' : null}
              onRestart={restart}
              onGuide={() => startGuided()}
              guideBusy={guidedBusy}
            />
          )}

          {!busy && showGuided && (
            <GuidedView
              session={guided.session}
              envelope={guided.envelope}
              catalogIds={guided.catalogIds}
              busy={guidedBusy}
              onAnswer={(outcome) => guidedCall(() => guidedAnswer(guided.session.session_id, outcome))}
              onConfirm={(proceed) => guidedCall(() => guidedConfirm(guided.session.session_id, proceed))}
              onEscalate={async () => {
                await guidedCall(() => guidedEscalate(guided.session.session_id))
                setBeat(7)
              }}
              onRestart={restart}
              onExit={result ? () => setBeat(3) : undefined}
            />
          )}
        </main>
      )}

      {result && view === 'engineer' && (
        <div className="panels">
          <EnrichmentPanel query={result.envelope.query} enrichment={result.debug.enrichment} />
          <TelemetryPanel
            meta={result.envelope.meta}
            cache={result.debug.cache}
            lastColdMs={lastColdMs}
          />
          {result.debug.evidence ? (
            <GroundingPanel
              evidence={result.debug.evidence}
              spans={markedSpans}
              steps={steps}
              activeStep={activeStep}
              setActiveStep={setActiveStep}
              replayedFrom={result.debug.grounding_from?.cold_run_query}
            />
          ) : (
            <section className="panel">
              <h2><span className="n">2</span> Where each step came from</h2>
              <p className="empty">
                No article was supplied with this request, so there is nothing to trace
                the steps to.
              </p>
            </section>
          )}
          <PlanPanel response={result.envelope.response} />
          <ResolverPanel
            resolutions={result.debug.resolutions}
            replayedFrom={result.debug.grounding_from?.cold_run_query}
          />
        </div>
      )}

      {view === 'customer' && (
        <footer className="pfoot">
          Smart Guided Troubleshooting Engine · grounded in the supplied Samsung SIIS
          knowledge base ·{' '}
          <a href={REPO_URL} target="_blank" rel="noreferrer">source repository</a>
        </footer>
      )}
    </div>
  )
}
