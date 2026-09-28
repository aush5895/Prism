import { useState } from 'react'
import { CATEGORY_WORDS, OpenButton } from './customer.jsx'

// GUIDED MODE — one step at a time.
//
// The backend owns everything that matters here: the order, the safety gate, and the
// handoff. This file only renders the session it is handed and forwards the customer's
// answer. It never decides what comes next, and it never writes warning text: anything
// shown at the safety gate is a sentence quoted from Samsung's own article, and when the
// article has none, the screen says so instead of filling the gap.

const OUTCOME_WORDS = {
  fixed: { label: 'Fixed it', cls: 'ok' },
  not_fixed: { label: 'Did not help', cls: 'no' },
  could_not_do: { label: 'Could not do it', cls: 'no' },
  skipped: { label: 'Skipped', cls: 'skip' },
  declined: { label: 'Declined', cls: 'skip' },
}

async function copyText(text) {
  try {
    await navigator.clipboard.writeText(text)
    return true
  } catch {
    return false
  }
}

function Progress({ session, actions }) {
  const byIndex = Object.fromEntries(session.attempts.map((a) => [a.index, a.outcome]))
  const current = session.current?.index
  return (
    <div className="g-progress">
      <div className="g-progress-label">
        {session.status === 'active' || session.status === 'awaiting_confirmation'
          ? <>Step <strong>{session.progress.position}</strong> of {session.progress.total}</>
          : <>{session.attempts.length} of {session.progress.total} steps answered</>}
      </div>
      <ol className="g-track" aria-label="Steps in this plan">
        {actions.map((a, i) => {
          const outcome = byIndex[i]
          const state = outcome ? OUTCOME_WORDS[outcome]?.cls
            : i === current ? 'now' : a.category === 'critical' ? 'crit' : ''
          return (
            <li
              key={i}
              className={`g-seg ${state || ''}`}
              title={`${i + 1}. ${a.actionName}${outcome ? ` — ${OUTCOME_WORDS[outcome]?.label}` : ''}`}
            />
          )
        })}
      </ol>
    </div>
  )
}

function StepCard({ action, index, catalogIds }) {
  const [openKey, setOpenKey] = useState(null)
  const words = CATEGORY_WORDS[action.category] || CATEGORY_WORDS.manual
  return (
    <article className="card cust-card g-step">
      <div className="cust-card-head">
        <span className="cust-num">{index + 1}</span>
        <h3>{action.actionName}</h3>
        <span className={`cust-badge ${words.cls}`}>{words.label}</span>
      </div>
      <p className="cust-why">{action.description}</p>
      {action.stepGroups.map((group, j) => (
        <div className="cust-group" key={j}>
          {action.stepGroups.length > 1 && (
            <div className="cust-alt">{j === 0 ? 'Try this' : 'Or, if that does not apply'}</div>
          )}
          <ol className="cust-steps">
            {group.steps.map((step, k) => <li key={k}>{step}</li>)}
          </ol>
          {group.actionableDeeplink && (
            <OpenButton
              deeplink={group.actionableDeeplink}
              validation={group.validationDeeplink}
              catalogId={catalogIds[group.actionableDeeplink.deeplink]}
              open={openKey === j}
              onToggle={() => setOpenKey((prev) => (prev === j ? null : j))}
            />
          )}
        </div>
      ))}
    </article>
  )
}

function SafetyGate({ current, busy, onConfirm }) {
  const gate = current.safety_gate
  return (
    <section className="card g-gate" aria-live="polite">
      <div className="eyebrow g-crit">Before you continue</div>
      <h2>Step {current.index + 1} is a last-resort step</h2>
      <p className="g-lede">
        <strong>{current.action.actionName}.</strong> It is classed as critical, a
        disruptive or irreversible operation, so we ask before showing it.
      </p>

      {gate.quotes.length > 0 ? (
        <div className="g-quotes">
          <div className="g-quotes-label">From Samsung&rsquo;s article</div>
          {gate.quotes.map((q) => <blockquote key={q}>{q}</blockquote>)}
        </div>
      ) : (
        // Same sentence whether or not the step is destructive: the lexicon not matching
        // is not proof the step is safe, so the screen does not say that it is.
        <p className="g-noquote">The supplied article gives no specific warning for this step.</p>
      )}

      <div className="g-actions">
        <button type="button" className="cta" disabled={busy} onClick={() => onConfirm(true)}>
          I understand, show me the step
        </button>
        <button type="button" className="g-btn" disabled={busy} onClick={() => onConfirm(false)}>
          Skip this step
        </button>
      </div>
    </section>
  )
}

function SupportPrompt({ current, busy, onEscalate, onAnswer }) {
  const after = current.critical_after
  return (
    <section className="card g-support">
      <div className="eyebrow">Your plan suggests talking to Samsung now</div>
      <p className="g-lede">
        {after > 0
          ? `The ${after} step${after === 1 ? '' : 's'} after this one ${after === 1 ? 'is a last-resort step' : 'are last-resort steps'}, so the plan puts support first.`
          : 'This is where the plan recommends contacting Samsung support.'}
        {' '}An agent gets a summary of everything you have already tried.
      </p>
      <div className="g-actions">
        <button type="button" className="cta" disabled={busy} onClick={onEscalate}>
          Connect me to an agent
        </button>
        <button type="button" className="g-btn" disabled={busy} onClick={() => onAnswer('skipped')}>
          Continue on my own
        </button>
      </div>
    </section>
  )
}

function Answer({ busy, onAnswer, onEscalate }) {
  return (
    <section className="g-answer">
      <div className="g-answer-q">Did this fix the problem?</div>
      <div className="g-actions">
        <button type="button" className="g-btn g-yes" disabled={busy} onClick={() => onAnswer('fixed')}>
          Yes, it&rsquo;s fixed
        </button>
        <button type="button" className="g-btn" disabled={busy} onClick={() => onAnswer('not_fixed')}>
          No, still happening
        </button>
        <button type="button" className="linkish" disabled={busy} onClick={() => onAnswer('could_not_do')}>
          I couldn&rsquo;t do this step
        </button>
      </div>
      <button type="button" className="linkish g-agent" disabled={busy} onClick={onEscalate}>
        Talk to an agent instead
      </button>
    </section>
  )
}

function Resolved({ session, actions, onRestart }) {
  const i = session.resolved_by
  return (
    <section className="card g-done">
      <div className="eyebrow g-ok">Fixed</div>
      <h2>Glad that worked</h2>
      <p className="g-lede">
        Step {i + 1} of {actions.length} fixed it: <strong>{actions[i]?.actionName}</strong>.
        {actions.length - i - 1 > 0 && ` You did not need the ${actions.length - i - 1} step${actions.length - i - 1 === 1 ? '' : 's'} after it${actions.slice(i + 1).some((a) => a.category === 'critical') ? ', including the last-resort ones' : ''}.`}
      </p>
      <button type="button" className="linkish" onClick={onRestart}>Describe a different problem</button>
    </section>
  )
}

function Handoff({ handoff, onRestart }) {
  const [copied, setCopied] = useState(null)
  return (
    <section className="card g-handoff">
      <div className="eyebrow">Agent handoff</div>
      <h2>An agent takes it from here</h2>
      <p className="g-lede">{handoff.headline}</p>
      <p className="g-reason">Why: {handoff.reason_label}.</p>

      {handoff.tried.length > 0 && (
        <div className="g-list">
          <div className="g-list-label">Step by step</div>
          <ul>
            {handoff.tried.map((t) => (
              <li key={t.step}>
                <span className="g-list-n">{t.step}</span>
                <span className="g-list-name">{t.action}</span>
                <span className={`g-chip ${OUTCOME_WORDS[t.outcome]?.cls || ''}`}>
                  {OUTCOME_WORDS[t.outcome]?.label || t.outcome}
                </span>
              </li>
            ))}
          </ul>
        </div>
      )}
      {handoff.not_tried.length > 0 && (
        <div className="g-list">
          <div className="g-list-label">Not yet tried</div>
          <ul>
            {handoff.not_tried.map((t) => (
              <li key={t.step}>
                <span className="g-list-n">{t.step}</span>
                <span className="g-list-name">{t.action}</span>
                {t.category === 'critical' && <span className="g-chip crit">Last resort</span>}
              </li>
            ))}
          </ul>
        </div>
      )}

      <div className="g-note">
        <div className="g-list-label">What the agent receives</div>
        <pre>{handoff.text}</pre>
        <button
          type="button"
          className="g-btn"
          onClick={async () => setCopied(await copyText(handoff.text))}
        >
          {copied === true ? 'Copied' : copied === false ? 'Select the text above to copy' : 'Copy summary'}
        </button>
      </div>
      <button type="button" className="linkish" onClick={onRestart}>Describe a different problem</button>
    </section>
  )
}

export function GuidedView({ session, envelope, catalogIds = {}, busy, onAnswer, onConfirm, onEscalate, onRestart, onExit }) {
  const context = envelope?.response?.contexts?.[0]
  const actions = context?.actions || []
  const current = session.current

  return (
    <div className="cust g-wrap">
      <header className="cust-head">
        <div className="eyebrow">Guided fix</div>
        <h2>{context?.title || 'No plan for this complaint'}</h2>
        {actions.length > 0 && <Progress session={session} actions={actions} />}
        {onExit && session.status !== 'resolved' && (
          <button type="button" className="linkish" onClick={onExit}>← See the whole plan</button>
        )}
      </header>

      {session.status === 'awaiting_confirmation' && current && (
        <SafetyGate current={current} busy={busy} onConfirm={onConfirm} />
      )}

      {session.status === 'active' && current && (
        <>
          <StepCard action={current.action} index={current.index} catalogIds={catalogIds} />
          {current.support_step
            ? <SupportPrompt current={current} busy={busy} onEscalate={onEscalate} onAnswer={onAnswer} />
            : <Answer busy={busy} onAnswer={onAnswer} onEscalate={onEscalate} />}
        </>
      )}

      {session.status === 'resolved' && (
        <Resolved session={session} actions={actions} onRestart={onRestart} />
      )}

      {(session.status === 'escalated' || session.status === 'no_plan') && session.handoff && (
        <Handoff handoff={session.handoff} onRestart={onRestart} />
      )}
    </div>
  )
}
