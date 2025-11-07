import { Reveal } from "../common/Reveal";

export function AgentsOverview() {
  const agents = [
    {
      title: "Discovery Agent",
      desc:
        "Continuously inventories systems to locate personal and sensitive data across sources, fixing visibility gaps.",
      emoji: "🛰️",
    },
    {
      title: "Cleaner Agent",
      desc:
        "Identifies PII and sensitive attributes, classifies content, and prepares data for remediation and audits.",
      emoji: "🧽",
    },
    {
      title: "Remediation Agent",
      desc:
        "Suggests anonymization, consent validation, or deletion; generates compliance reports and monitors posture.",
      emoji: "🛡️",
    },
  ];

  return (
    <div className="container-max py-16 content-auto" id="agents">
      <Reveal className="text-center" as="div">
        <h2 className="text-2xl sm:text-3xl font-bold text-slate-900">Our agents</h2>
        <p className="mt-3 text-slate-600 max-w-2xl mx-auto">
          Modular, AI-driven roles that work together to keep your data compliant.
        </p>
      </Reveal>
  <div className="mt-10 grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-4 sm:gap-6">
        {agents.map((a, i) => (
          <Reveal key={a.title} delayMs={i * 110}>
            <div className="rounded-xl border border-slate-200 bg-white/80 p-6 shadow-sm transition-transform duration-300 hover:-translate-y-0.5">
              <div className="text-3xl">{a.emoji}</div>
              <h3 className="mt-3 font-semibold text-slate-900">{a.title}</h3>
              <p className="mt-2 text-sm text-slate-600">{a.desc}</p>
            </div>
          </Reveal>
        ))}
      </div>
    </div>
  );
}
