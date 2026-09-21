import { useEffect, useMemo, useState } from 'react'
import { API_BASE, fetchSamples, troubleshoot } from './api.js'
import {
  EnrichmentPanel, GroundingPanel, PlanPanel, ResolverPanel, TelemetryPanel,
} from './panels.jsx'

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

  useEffect(() => {
    fetchSamples()
      .then((data) => {
        setSamples(data.samples)
        if (data.samples.length) {
          setSelected(data.samples[0].id)
          setQuery(data.samples[0].query)
        }
      })
      .catch((e) => setError(`Cannot reach the API at ${API_BASE}. Is it running? (${e.message})`))
  }, [])

  const sample = samples.find((s) => s.id === selected)

  function pickSample(id) {
    const next = samples.find((s) => s.id === id)
    setSelected(id)
    if (next) setQuery(next.query)
    setResult(null)
    setLastColdMs(null)
  }

  async function run() {
    if (!sample) return
    setBusy(true)
    setError(null)
    try {
      const data = await troubleshoot(query, sample.siis_response)
      if (!data.envelope.meta.cache_hit) setLastColdMs(data.envelope.meta.latency_ms)
      setResult(data)
    } catch (e) {
      setError(e.message)
    } finally {
      setBusy(false)
    }
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

  const markedSpans = useMemo(
    () => steps.filter((s) => s.span).map((s) => ({ ...s.span, key: s.key })),
    [steps],
  )

  return (
    <div className="app">
      <header className="top">
        <h1>Smart Guided Troubleshooting Engine</h1>
        <span className="sub">Samsung PRISM GenAI Hackathon · Theme 02</span>
        <span className="spacer" />
        <span className="sub mono">{API_BASE}</span>
      </header>

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
        <button onClick={run} disabled={busy || !sample}>
          {busy ? 'Running…' : 'Run'}
        </button>
        <button className="ghost" onClick={run} disabled={busy || !sample || !result}>
          Run again (cache)
        </button>
      </div>
      <p className="hint">
        Run once to see the cold path, then reword the complaint and run again: the same
        article with a different phrasing should be answered from cache.
      </p>

      {error && <div className="err">{error}</div>}

      {result && (
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
            />
          ) : (
            <section className="panel">
              <h2><span className="n">2</span> Grounding</h2>
              <p className="empty">
                Served from cache — grounding and extraction did not run. Change the
                article to see this panel populated.
              </p>
            </section>
          )}
          <PlanPanel response={result.envelope.response} />
          <ResolverPanel resolutions={result.debug.resolutions} />
        </div>
      )}
    </div>
  )
}
