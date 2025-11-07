import { Reveal } from "../common/Reveal";

export function Steps() {
  const steps = [
    { title: 'Discover', desc: 'Continuously inventory data across apps, logs, DBs, and cloud.' , emoji: '🧭' },
    { title: 'Classify', desc: 'Detect PII and sensitive categories with AI labeling.', emoji: '🔎' },
    { title: 'Mitigate', desc: 'Anonymize, minimize, and enforce consent governance.', emoji: '🛡️' },
    { title: 'Monitor', desc: 'Continuous checks and reports mapped to GDPR.', emoji: '📈' },
  ];
  return (
    <section className="container-max py-16 content-auto">
      <Reveal className="text-center" as="div">
        <h2 className="text-2xl sm:text-3xl font-bold text-slate-900">How it works</h2>
        <p className="mt-3 text-slate-600 max-w-2xl mx-auto">Simple, proactive steps to keep your data compliant.</p>
      </Reveal>
  <div className="mt-10 grid grid-cols-1 sm:grid-cols-2 md:grid-cols-4 gap-4 sm:gap-6">
        {steps.map((s, i) => (
          <Reveal key={s.title} delayMs={i * 100}>
            <div className="relative rounded-xl border border-slate-200 bg-white/80 p-6 shadow-sm transition-transform duration-300 hover:-translate-y-0.5">
              <div className="text-3xl">{s.emoji}</div>
              <div className="mt-3 flex items-baseline gap-2">
                <span className="text-xs text-slate-500">Step {i + 1}</span>
                <h3 className="font-semibold text-slate-900">{s.title}</h3>
              </div>
              <p className="mt-2 text-sm text-slate-600">{s.desc}</p>
            </div>
          </Reveal>
        ))}
      </div>
    </section>
  );
}
