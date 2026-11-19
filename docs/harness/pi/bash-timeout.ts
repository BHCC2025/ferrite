/**
 * bash-timeout.ts — default and maximum run time for the model's shell commands.
 *
 * Pi's bash tool takes an optional `timeout` (seconds) with no default, so a
 * command that never exits (a test script kept alive by a timer, a server
 * started in the foreground, a prompt waiting for input) blocks the session
 * until someone presses Esc. This extension:
 *   - sets a default timeout when the model gives none,
 *   - caps a timeout the model asks for,
 *   - adds a short note to the result when a command times out, so the model
 *     knows what happened and what to do next.
 * It applies to every model and every project (global extension).
 *
 * Settings: ~/.pi/agent/bash-timeout.json (optional), re-read on every call:
 *   { "defaultSeconds": 120, "maxSeconds": 600, "enabled": true }
 * Commands the user types with ! or !! are not affected.
 */

import { readFileSync } from "node:fs";
import { homedir } from "node:os";
import { join } from "node:path";
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { isToolCallEventType } from "@earendil-works/pi-coding-agent";

const CONFIG = join(homedir(), ".pi", "agent", "bash-timeout.json");

function settings() {
	let c: { defaultSeconds?: number; maxSeconds?: number; enabled?: boolean } = {};
	try {
		c = JSON.parse(readFileSync(CONFIG, "utf8"));
	} catch {
		// no file or unreadable: defaults
	}
	const pos = (v: unknown, d: number) => (typeof v === "number" && Number.isFinite(v) && v > 0 ? v : d);
	const def = pos(c.defaultSeconds, 120);
	const max = Math.max(def, pos(c.maxSeconds, 600));
	return { enabled: c.enabled !== false, def, max };
}

export default function (pi: ExtensionAPI) {
	pi.on("tool_call", async (event) => {
		if (!isToolCallEventType("bash", event)) return;
		const s = settings();
		if (!s.enabled) return;
		const t = event.input.timeout;
		if (typeof t !== "number" || !Number.isFinite(t) || t <= 0) event.input.timeout = s.def;
		else if (t > s.max) event.input.timeout = s.max;
	});

	pi.on("tool_result", async (event) => {
		if (event.toolName !== "bash" || !Array.isArray(event.content)) return;
		const text = event.content.map((c: any) => (c?.type === "text" ? c.text : "")).join("");
		if (!/Command timed out after [\d.]+ seconds/.test(text)) return;
		const s = settings();
		const note =
			`\n\n[The command was stopped at the time limit. If it needs longer, pass \`timeout\` (seconds, up to ${s.max}). ` +
			`If it never exits on its own (a watcher, a server, a script kept alive by a timer or an open handle), ` +
			`change it so it ends: run it in the background, add an explicit exit, or use a bounded check.]`;
		return { content: [...event.content, { type: "text", text: note }] };
	});
}
