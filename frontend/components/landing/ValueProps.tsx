export function ValueProps() {
  const items = [
    { title: 'EU-first compliance', desc: 'Designed around GDPR principles: lawfulness, purpose limitation, minimization, and accountability.', emoji: '🇪🇺' },
    { title: 'Nordic identity ready', desc: 'Built with BankID, MitID, and Suomi.fi contexts in mind for seamless identity-aware workflows.', emoji: '🧩' },
    { title: 'Continuous by default', desc: 'Move from manual audits to ongoing monitoring with alerts and clear reports.', emoji: '⏱️' },
  ];
  return (
    <section className="container-max py-16">
      <h2 className="text-2xl sm:text-3xl font-bold text-slate-900 text-center">Why it fits the Nordics</h2>
      <div className="mt-8 grid grid-cols-1 md:grid-cols-3 gap-6">
        {items.map((v) => (
          <div key={v.title} className="rounded-xl border border-slate-200 bg-white/70 backdrop-blur p-6 shadow-sm">
            <div className="text-3xl">{v.emoji}</div>
            <h3 className="mt-3 font-semibold text-slate-900">{v.title}</h3>
            <p className="mt-2 text-sm text-slate-600">{v.desc}</p>
          </div>
        ))}
      </div>
    </section>
  );
}
