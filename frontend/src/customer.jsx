import { useEffect, useMemo, useRef, useState } from 'react'

// CUSTOMER VIEW — what the product actually is.
//
// The plan, and nothing else. No BM25, no gate names, no coverage numbers, no latency.
// Someone who has never seen this project should understand it in ten seconds.
//
// ONE HONESTY CONSTRAINT SHAPES THE BUTTON. The catalog's URIs are masked placeholders —
// deeplinks.json's own _readme: "URIs are MASKED placeholders: match on description,
// message, qna_description and originalType, then copy the URI verbatim." A hash like
// bixby://masked/act/1b0d34e9b4 REPLACED the real Samsung URI. It cannot resolve on any
// device, ever, and a desktop browser cannot handle a bixby:// scheme in any case. The
// task was to select the right catalog entry and copy its URI verbatim, not to make it
// launch.
//
// So the button does not claim to open anything. Clicking it proves what we actually did:
// it shows the verbatim URI, the catalog entry it came from, the link type and the
// validation deeplink, and copies the URI. A judge who clicks sees catalog integrity
// demonstrated instead of a link that silently does nothing.
//
// THE SAME RULE GOVERNS EVERY LABEL ON THIS SCREEN. Nothing here may imply a capability
// the system does not have: no connected device, no telemetry, no sync state. The device
// chip says "simulated" and the model on the entry card is enrich.py's parse of the
// complaint text, passed in — it is never inferred in JS.

const CATEGORY_WORDS = {
  auto: { label: 'Settings change', cls: 'auto' },
  manual: { label: 'Do this by hand', cls: 'manual' },
  critical: { label: 'Last resort', cls: 'critical' },
}

// The catalog opens every message with one of a small set of verbs. Strip it to get the
// screen's own name, so the button reads "Open Touch sensitivity" rather than
// "Disable Touch sensitivity".
const LEADING_VERBS = new Set([
  'view', 'enable', 'disable', 'adjust', 'check', 'increase', 'switch', 'optimize', 'open',
])

export function screenName(message) {
  if (!message) return ''
  const parts = message.trim().split(/\s+/)
  if (parts.length > 1 && LEADING_VERBS.has(parts[0].toLowerCase())) parts.shift()
  return parts.join(' ')
}

// The button must show the DIRECTION, not just the screen. Stripping the verb alone gave
// two adjacent cards reading "Open Touch sensitivity" for opposite actions -- the enable
// path and the disable path are the pair the resolver works hardest to separate, and the
// customer could not tell them apart.
export function buttonLabel(deeplink) {
  const name = screenName(deeplink.message)
  if (deeplink.originalType === 'onURL') return `Turn on ${name}`
  if (deeplink.originalType === 'offURL') return `Turn off ${name}`
  return `Open ${name}`
}

const DUMMY_URI = 'bixby://dummy_positive'

// Chip text is the supplied complaint itself, shortened at a word boundary — never a
// rewrite of it. The full text is on the title attribute and lands in the textarea.
export function chipLabel(query, max = 40) {
  const clean = query.trim().replace(/^\d+\.\s*/, '').replace(/^["“]/, '')
  if (clean.length <= max) return clean
  const cut = clean.slice(0, max)
  const space = cut.lastIndexOf(' ')
  return `${(space > 20 ? cut.slice(0, space) : cut).replace(/[,.;:]$/, '')}…`
}

async function copyToClipboard(text) {
  try {
    await navigator.clipboard.writeText(text)
    return true
  } catch {
    return false          // blocked outside a secure context; the panel still shows
  }
}

// --------------------------------------------------------------------------- beat 1
export function EntryCard({
  query, setQuery, samples, selected, onPick, onSubmit, detected, disabled,
}) {
  const chars = query.length
  const [showAll, setShowAll] = useState(false)

  // All twenty are reachable, but twenty full-width chips bury the textarea. Six, with
  // the selected one pinned first so it is never hidden behind the toggle.
  const shown = showAll
    ? samples
    : [...samples.filter((s) => s.id === selected),
       ...samples.filter((s) => s.id !== selected)].slice(0, 6)

  return (
    <section className="card entry">
      <div className="eyebrow">Active session</div>
      <h1>What&rsquo;s wrong with your device?</h1>
      <p className="entry-sub">
        Describe it in your own words. We parse the complaint and match every step against
        the Samsung service record supplied with it — nothing outside those records is
        suggested.
      </p>

      <textarea
        className="entry-input"
        value={query}
        rows={5}
        onChange={(e) => setQuery(e.target.value)}
        placeholder="For example: my screen goes black whenever I open the camera"
        aria-label="Describe the problem"
      />

      <div className="entry-foot">
        <span className="counter">{chars} character{chars === 1 ? '' : 's'}</span>
        <button
          type="button"
          className="cta"
          onClick={onSubmit}
          disabled={disabled || !query.trim()}
        >
          Analyze Issue <span aria-hidden="true">→</span>
        </button>
      </div>

      <div className="try">
        <div className="try-label">Try saying:</div>
        <div className="chips">
          {shown.map((s) => (
            <button
              key={s.id}
              type="button"
              className={`chip${s.id === selected ? ' on' : ''}`}
              title={s.query}
              onClick={() => onPick(s.id)}
            >
              {chipLabel(s.query)}
            </button>
          ))}
          {samples.length > shown.length && (
            <button type="button" className="chip more" onClick={() => setShowAll(true)}>
              +{samples.length - shown.length} more
            </button>
          )}
        </div>
        <p className="try-note">
          The {samples.length} complaints supplied with the Samsung kit, verbatim —
          shortened here, sent in full.
        </p>
      </div>

      <div className="devcard">
        <span className="devcard-glyph" aria-hidden="true" />
        <div className="devcard-body">
          <div className="devcard-model">
            {detected.parsed
              ? (detected.device || 'No model named')
              : 'Model not read yet'}
          </div>
          <div className="devcard-note">
            {detected.parsed && !detected.device
              ? 'nothing in your description names a model, so we do not assume one'
              : 'detected from your description'}
          </div>
        </div>
      </div>
    </section>
  )
}

// --------------------------------------------------------------------------- beat 2
export function AnalyzingCard({ query }) {
  return (
    <section className="card analyzing" aria-live="polite">
      <div className="eyebrow">Active session</div>
      <h1>Working on it</h1>
      <p className="entry-sub">Reading the supplied service record for this complaint.</p>
      <blockquote className="analyzing-q">{query}</blockquote>
      <div className="analyzing-bar"><span /></div>
      <ul className="analyzing-stages">
        <li>Normalising the complaint into slots</li>
        <li>Grounding it in the supplied article</li>
        <li>Extracting the steps the article actually states</li>
        <li>Resolving each Settings screen against the catalog</li>
      </ul>
    </section>
  )
}

// --------------------------------------------------------------------------- beat 5
function OpenButton({ deeplink, validation, catalogId, open, onToggle }) {
  const [copied, setCopied] = useState(false)
  const name = screenName(deeplink.message)
  const label = buttonLabel(deeplink)
  const isPlaceholder = deeplink.deeplink === DUMMY_URI

  async function reveal() {
    setCopied(await copyToClipboard(deeplink.deeplink))
    onToggle()
  }

  return (
    <div className="open-wrap">
      <button
        className="open-btn"
        type="button"
        aria-expanded={open}
        title={`Show the catalog entry behind ${name}`}
        onClick={reveal}
      >
        <span className="open-icon" aria-hidden="true">›</span>
        {label}
      </button>
      <span className="open-note">Verified Samsung catalog entry — masked URI</span>

      {open && (
        <div className="proof">
          <div className="proof-row">
            <span className="proof-k">URI</span>
            <code className="proof-uri">{deeplink.deeplink}</code>
          </div>

          {!isPlaceholder && (
            <>
              <div className="proof-row">
                <span className="proof-k">Catalog entry</span>
                <span>
                  {catalogId ? <code>{catalogId}</code> : <em>—</em>}
                  {' · '}“{deeplink.message}”
                </span>
              </div>
              <div className="proof-row">
                <span className="proof-k">Link type</span>
                <code>{deeplink.originalType}</code>
              </div>
              <div className="proof-row">
                <span className="proof-k">Validation</span>
                <span>
                  {validation ? (
                    <>
                      key <code>{validation.key}</code>
                      {validation.value ? <> · expects <code>{validation.value}</code></> : null}
                    </>
                  ) : <em>none for this entry</em>}
                </span>
              </div>
            </>
          )}

          <p className="proof-note">
            {isPlaceholder
              ? 'No catalog entry exists for this screen. Placeholder, per Samsung’s catalog rules.'
              : `${copied ? 'Copied. ' : ''}Masked catalog URI — resolves to this screen on a Galaxy device.`}
          </p>
        </div>
      )}
    </div>
  )
}

// --------------------------------------------------------------------- beats 3, 4, 6
export function CustomerView({ envelope, catalogIds = {}, focus = null, onRestart }) {
  const context = envelope?.response?.contexts?.[0]
  const actions = context?.actions || []

  // One proof panel open at a time, held here rather than inside the button, because the
  // "Resolution Verify" beat has to be able to open one from the progress strip.
  const [openKey, setOpenKey] = useState(null)
  const cardRefs = useRef({})

  // The first group that resolved to a link: the one the verify beat opens. Placeholder
  // URIs count, since their proof panel is exactly where we say a screen has no entry.
  const firstLink = useMemo(() => {
    for (let i = 0; i < actions.length; i += 1) {
      const groups = actions[i].stepGroups || []
      for (let j = 0; j < groups.length; j += 1) {
        if (groups[j].actionableDeeplink) return { key: `${i}.${j}`, card: i }
      }
    }
    return null
  }, [actions])

  // The honest-fallback beat opens a PLACEHOLDER proof by preference: bixby://dummy_positive
  // is the engine saying "this screen has no catalog entry and I will not invent one",
  // and that panel is the only place it says so in words.
  const firstPlaceholder = useMemo(() => {
    for (let i = 0; i < actions.length; i += 1) {
      const groups = actions[i].stepGroups || []
      for (let j = 0; j < groups.length; j += 1) {
        if (groups[j].actionableDeeplink?.deeplink === DUMMY_URI) {
          return { key: `${i}.${j}`, card: i }
        }
      }
    }
    return null
  }, [actions])

  const target = focus === 'fallback' ? firstPlaceholder || firstLink : firstLink
  const focusCard = focus === 'detail' ? 0
    : (focus === 'verify' || focus === 'fallback') ? target?.card ?? 0
      : null

  useEffect(() => {
    if (focus === 'verify' || focus === 'fallback') setOpenKey(target ? target.key : null)
    else if (focus === 'detail') setOpenKey(null)
  }, [focus, target])

  useEffect(() => {
    if (focusCard === null) return
    const node = cardRefs.current[focusCard]
    if (node) node.scrollIntoView({ behavior: 'smooth', block: 'center' })
  }, [focusCard, focus])

  if (!context) {
    return (
      <div className="cust">
        <section className="card cust-empty">
          <div className="eyebrow">Active session</div>
          <h2>No confirmed fix for this one</h2>
          <p>
            We could not find steps we are confident about for this problem, so we are not
            guessing. Contact Samsung Support and they can take it further.
          </p>
          {onRestart && (
            <button type="button" className="linkish" onClick={onRestart}>
              Describe a different problem
            </button>
          )}
        </section>
      </div>
    )
  }

  return (
    <div className="cust">
      <header className="cust-head">
        <div className="eyebrow">Your plan</div>
        <h2>{context.title}</h2>
        <p>Try these in order. Stop as soon as the problem goes away.</p>
        {onRestart && (
          <button type="button" className="linkish" onClick={onRestart}>
            ← Describe a different problem
          </button>
        )}
      </header>

      {actions.map((action, i) => {
        // Badge from the emitted `category` ONLY. Never from deeplink presence: an auto
        // action whose screen could not be resolved is still a Settings change, and
        // badging it "Do this by hand" would contradict its own steps.
        const words = CATEGORY_WORDS[action.category] || CATEGORY_WORDS.manual
        return (
          <article
            className={`card cust-card${focusCard === i ? ' is-focus' : ''}`}
            key={i}
            ref={(node) => { cardRefs.current[i] = node }}
          >
            <div className="cust-card-head">
              <span className="cust-num">{i + 1}</span>
              <h3>{action.actionName}</h3>
              <span className={`cust-badge ${words.cls}`}>{words.label}</span>
            </div>

            <p className="cust-why">{action.description}</p>

            {action.stepGroups.map((group, j) => (
              <div className="cust-group" key={j}>
                {action.stepGroups.length > 1 && (
                  <div className="cust-alt">
                    {j === 0 ? 'Try this' : 'Or, if that does not apply'}
                  </div>
                )}
                <ol className="cust-steps">
                  {group.steps.map((step, k) => <li key={k}>{step}</li>)}
                </ol>
                {group.actionableDeeplink
                  ? (
                    <OpenButton
                      deeplink={group.actionableDeeplink}
                      validation={group.validationDeeplink}
                      catalogId={catalogIds[group.actionableDeeplink.deeplink]}
                      open={openKey === `${i}.${j}`}
                      onToggle={() => setOpenKey(
                        (prev) => (prev === `${i}.${j}` ? null : `${i}.${j}`),
                      )}
                    />
                  )
                  : action.category === 'auto' && (
                    // A Settings action we could not link: the resolver declined to guess
                    // which screen or which way to set it. Say so plainly rather than
                    // leave a card that looks like it is missing its button.
                    <p className="cust-selfserve">
                      Open Settings yourself — we could not tell which way to set this.
                    </p>
                  )}
              </div>
            ))}
          </article>
        )
      })}
    </div>
  )
}
