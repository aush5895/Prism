// CUSTOMER VIEW — what the product actually is.
//
// The plan, and nothing else. No BM25, no gate names, no catalog ids, no coverage
// numbers, no latency. Someone who has never seen this project should understand it in
// ten seconds. Everything an engineer needs is one toggle away in the engineer view.

// Plain words for the schema's three categories. The API still emits auto|manual|critical
// (that enum is Samsung's and is not ours to change); this is presentation only.
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

function OpenButton({ deeplink }) {
  const name = screenName(deeplink.message)
  return (
    <div className="open-wrap">
      <button
        className="open-btn"
        type="button"
        title={`Opens ${name} on your Galaxy device`}
        onClick={(e) => e.currentTarget.blur()}
      >
        <span className="open-icon" aria-hidden="true">›</span>
        Open {name}
      </button>
      <span className="open-note">Opens this screen on your phone</span>
    </div>
  )
}

export function CustomerView({ envelope }) {
  const context = envelope?.response?.contexts?.[0]

  if (!context) {
    return (
      <div className="cust">
        <div className="cust-empty">
          <h2>No confirmed fix for this one</h2>
          <p>
            We could not find steps we are confident about for this problem, so we are not
            guessing. Contact Samsung Support and they can take it further.
          </p>
        </div>
      </div>
    )
  }

  return (
    <div className="cust">
      <header className="cust-head">
        <h2>{context.title}</h2>
        <p>Try these in order. Stop as soon as the problem goes away.</p>
      </header>

      {context.actions.map((action, i) => {
        const words = CATEGORY_WORDS[action.category] || CATEGORY_WORDS.manual
        return (
          <article className="cust-card" key={i}>
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
                {group.actionableDeeplink && <OpenButton deeplink={group.actionableDeeplink} />}
              </div>
            ))}
          </article>
        )
      })}
    </div>
  )
}
