"use client";
import { useState } from "react";

export function ChatbotPanel() {
	const [messages] = useState<{ role: "user" | "assistant"; content: string }[]>([
		{ role: "assistant", content: "Hi! I'll help you interpret compliance results soon." },
	]);

		return (
			<div className="flex flex-col h-full border-l border-slate-200 bg-white/80">
			<div className="h-14 flex items-center px-4 border-b border-slate-200">
				<h2 className="font-semibold text-sm text-brand-700">Privacy Copilot</h2>
			</div>
			<div className="flex-1 overflow-y-auto p-4 space-y-3">
				{messages.map((m, i) => (
					<div
						key={i}
						className={"rounded-md px-3 py-2 text-sm max-w-[80%] " + (m.role === "assistant" ? "bg-brand-600/10 text-brand-800" : "bg-brand-600 text-white ml-auto")}
					>
						{m.content}
					</div>
				))}
			</div>
			<div className="p-3 border-t border-slate-200">
				<form className="flex gap-2" onSubmit={e => e.preventDefault()}>
					<input
						disabled
						placeholder="Chat coming soon..."
						className="flex-1 rounded-md border border-slate-300 bg-slate-50 px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-brand-400 disabled:opacity-60"
					/>
					<button
						type="submit"
						disabled
						className="rounded-md bg-brand-600 text-white px-4 py-2 text-sm font-medium disabled:opacity-50"
					>
						Send
					</button>
				</form>
			</div>
		</div>
	);
}
