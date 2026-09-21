// The five panels. Every value rendered here is read verbatim from the API response —
// no scoring, ordering, resolving or span arithmetic happens in the browser.

const nf = (n, d = 1) => (n === null || n === undefined ? '—' : Number(n).toFixed(d))

// ----------------------------------------------------------------- 1. enrichment
export function EnrichmentPanel({ query, enrichment }) {
  const slots = [
    ['device', enrichment?.device],
    ['domain', enrichment?.domain],
    ['symptoms', enrichment?.symptoms?.length ? enrichment.symptoms.join(', ') : null],
  ]
  return (
    <section className="panel">
      <h2><span className="n">1</span> Query &amp; enrichment</h2>
      <div className="raw-complaint">{query}</div>
      <div className="slots">
        {slots.map(([k, v]) => (
          <div className="slot" key={k}>
            <span className="k">{k}</span>
            <span className={v ? 'v' : 'v none'}>{v || 'not detected'}</span>
          </div>
        ))}
      </div>
      <div className="canonical">
        <span className="k">canonical query (the cache key)</span>
        <code className="v mono">{enrichment?.canonical || '—'}</code>
      </div>
    </section>
  )
}

// ----------------------------------------------------------------- 2. grounding
// Renders the article once, splicing in <mark> at the span offsets the backend verified.
// Overlapping spans are merged so the text is never duplicated or dropped.
export function GroundingPanel({ evidence, spans, steps, activeStep, setActiveStep }) {
  const text = evidence?.text || ''
  const located = spans.filter((s) => s.start !== null && s.start !== undefined)

  const pieces = []
  let cursor = 0
  const sorted = [...located].sort((a, b) => a.start - b.start)
  sorted.forEach((span, i) => {
    if (span.start < cursor) return          // already covered by an earlier mark
    if (span.start > cursor) pieces.push(text.slice(cursor, span.start))
    const isActive = activeStep === span.key
    pieces.push(
      <mark
        key={`m${i}`}
        className={isActive ? 'active' : ''}
        onMouseEnter={() => setActiveStep(span.key)}
        onMouseLeave={() => setActiveStep(null)}
      >
        {text.slice(span.start, span.end)}
      </mark>,
    )
    cursor = span.end
  })
  if (cursor < text.length) pieces.push(text.slice(cursor))

  const unlocated = spans.length - located.length
  return (
    <section className="panel">
      <h2>
        <span className="n">2</span> Grounding
        <span className="note">
          {located.length}/{spans.length} steps traced to the article
          {unlocated ? ` · ${unlocated} not locatable` : ''}
        </span>
      </h2>

      <div className="article">{pieces}</div>

      <ul className="steplist" style={{ marginTop: 12 }}>
        {steps.map((s) => (
          <li
            key={s.key}
            className={`${activeStep === s.key ? 'active' : ''} ${s.span ? '' : 'unlocated'}`}
            onMouseEnter={() => setActiveStep(s.key)}
            onMouseLeave={() => setActiveStep(null)}
          >
            {s.text}
            <span className="who">
              {s.span
                ? `${s.span.provenance} ${nf(s.span.confidence, 2)}`
                : 'no span'}
            </span>
          </li>
        ))}
      </ul>
    </section>
  )
}

// ----------------------------------------------------------------- 3. resolver
export function ResolverPanel({ resolutions }) {
  if (!resolutions?.length) return null
  return (
    <section className="panel wide">
      <h2>
        <span className="n">3</span> Resolver — accepted vs rejected
        <span className="note">gates [2] polarity · [3] scope · [4] target concept · [5] margin</span>
      </h2>
      {resolutions.map((r, i) => (
        <div className="resolution" key={i}>
          <div className="head">
            <span>{r.action}</span>
            {r.decision === 'exact' && <span className="tag accept">accepted {r.catalog_id}</span>}
            {r.decision === 'dummy_positive' && <span className="tag dummy">dummy_positive</span>}
            {r.decision === 'none' && <span className="tag none">null</span>}
            <span className="reason mono">{r.reason}</span>
          </div>
          {(r.rejected?.length > 0 || r.decision === 'exact') && (
            <table className="verdict-table">
              <thead>
                <tr>
                  <th>Catalog entry</th>
                  <th>Type</th>
                  <th className="num">BM25</th>
                  <th className="num">Coverage</th>
                  <th>Read as</th>
                  <th>Verdict</th>
                </tr>
              </thead>
              <tbody>
                {r.decision === 'exact' && (
                  <tr className="accepted">
                    <td><strong>{r.accepted_message || r.catalog_id}</strong></td>
                    <td className="mono">{r.accepted_type || '—'}</td>
                    <td className="num">—</td>
                    <td className="num">—</td>
                    <td className="mono">{r.reason?.replace('intent=', '') || '—'}</td>
                    <td><span className="tag accept">ACCEPTED</span></td>
                  </tr>
                )}
                {r.rejected?.map((c, j) => (
                  <tr className="rejected" key={j}>
                    <td>{c.message} <span className="mono" style={{ opacity: 0.55 }}>{c.catalog_id}</span></td>
                    <td className="mono">{c.original_type}</td>
                    <td className="num">{nf(c.bm25, 3)}</td>
                    <td className="num">{nf(c.coverage, 2)}</td>
                    <td className="mono">{c.intent}</td>
                    <td><span className="tag reject">{c.verdict.replace('reject:', '')}</span></td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </div>
      ))}
    </section>
  )
}

// ----------------------------------------------------------------- 4. plan
export function PlanPanel({ response }) {
  const context = response?.contexts?.[0]
  if (!context) {
    return (
      <section className="panel">
        <h2><span className="n">4</span> Plan</h2>
        <p className="empty">No plan — the pipeline fell back rather than guess.</p>
      </section>
    )
  }
  return (
    <section className="panel">
      <h2>
        <span className="n">4</span> Plan
        <span className="note">tier order · score {context.score}</span>
      </h2>
      <div style={{ marginBottom: 12 }}>
        <div style={{ fontSize: 17, fontWeight: 700 }}>{context.title}</div>
        <div style={{ color: 'var(--dim)', fontSize: 14 }}>{context.goal}</div>
      </div>
      {context.actions.map((a, i) => (
        <div className="action" key={i}>
          <div className="head">
            <span className="name">{a.actionName}</span>
            <span className={`tag ${a.category}`}>{a.category}</span>
            <span className="tier">#{i + 1}</span>
          </div>
          <div className="desc">{a.description}</div>
          {a.stepGroups.map((g, j) => (
            <div className="group" key={j}>
              <ol style={{ margin: '6px 0 0', paddingLeft: 22, fontSize: 14 }}>
                {g.steps.map((s, k) => <li key={k}>{s}</li>)}
              </ol>
              {g.actionableDeeplink ? (
                <div className="deeplink">
                  <div><strong>{g.actionableDeeplink.message}</strong> <span className="tag none">{g.actionableDeeplink.originalType}</span></div>
                  <div className="uri mono">{g.actionableDeeplink.deeplink}</div>
                  {g.validationDeeplink && (
                    <span className="val mono">
                      validation · {g.validationDeeplink.key}
                      {g.validationDeeplink.value ? ` = ${g.validationDeeplink.value}` : ''}
                    </span>
                  )}
                </div>
              ) : (
                <div className="deeplink null">no deeplink — null by design</div>
              )}
            </div>
          ))}
        </div>
      ))}
    </section>
  )
}

// ----------------------------------------------------------------- 5. telemetry
export function TelemetryPanel({ meta, cache, lastColdMs }) {
  const stages = Object.entries(meta?.stage_latency_ms || {})
  const peak = Math.max(1, ...stages.map(([, v]) => v))
  const hit = meta?.cache_hit
  const speedup = hit && lastColdMs ? Math.round(lastColdMs / Math.max(meta.latency_ms, 0.01)) : null

  return (
    <section className="panel">
      <h2><span className="n">5</span> Telemetry</h2>

      <div className={hit ? 'cachebar hit' : 'cachebar miss'}>
        {hit
          ? `CACHE HIT (${cache?.tier || 'L?'}) — no LLM call${speedup ? ` · ${speedup}× faster than the cold run` : ''}`
          : 'CACHE MISS — full pipeline, one LLM call'}
      </div>

      <div className="tiles" style={{ marginTop: 14 }}>
        <div className={`tile ${hit ? 'cold' : 'hot'}`}>
          <div className="k">latency</div>
          <div className="v">{nf(meta?.latency_ms, 1)}<span style={{ fontSize: 15 }}> ms</span></div>
        </div>
        <div className="tile">
          <div className="k">cost</div>
          <div className="v">${nf(meta?.cost_usd, 5)}</div>
        </div>
        <div className="tile">
          <div className="k">model</div>
          <div className="v" style={{ fontSize: 15, fontWeight: 600 }}>{meta?.model || '—'}</div>
        </div>
        {cache?.similarity != null && (
          <div className="tile">
            <div className="k">similarity</div>
            <div className="v">{nf(cache.similarity, 3)}</div>
          </div>
        )}
      </div>

      {meta?.fallback && (
        <p style={{ color: 'var(--warn)', marginTop: 12 }}>fallback: <code>{meta.fallback}</code></p>
      )}

      <div className="stages">
        {stages.map(([name, ms]) => (
          <div className="stage" key={name}>
            <span className="n mono">{name}</span>
            <span className="bar" style={{ width: `${Math.max(2, (ms / peak) * 55)}%` }} />
            <span className="ms">{nf(ms, 1)} ms</span>
          </div>
        ))}
      </div>

      {cache?.source_query && (
        <p style={{ color: 'var(--dim)', fontSize: 13, marginTop: 12 }}>
          served from a plan originally built for: <em>{cache.source_query}</em>
        </p>
      )}
    </section>
  )
}
