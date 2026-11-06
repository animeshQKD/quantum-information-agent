"""
mcp_client_check.py - test the MCP server (mcp_server.py) the way Claude Desktop or Cursor use it: a real
MCP client over streamable HTTP. Checks numbers against closed-form results, images, sessions, errors
and the security limits.

    python app.py                                   (in another terminal; then)
    python mcp_client_check.py                      (local: http://127.0.0.1:7860/gradio_api/mcp/)
    python mcp_client_check.py https://animesh1986-quantum-information-agent.hf.space/gradio_api/mcp/

On the Space every tool name has the prefix 'quantum_information_agent_'; it is detected automatically.
About 45 tool calls (the Space allows 120 per minute). Needs the 'mcp' package (pip install "gradio[mcp]").
"""
import asyncio
import base64
import json
import math
import sys
import time

from mcp import ClientSession
try:                                                    # newer mcp versions; older ones have the old name
    from mcp.client.streamable_http import streamable_http_client as http_client
except ImportError:
    from mcp.client.streamable_http import streamablehttp_client as http_client

LOCAL = "http://127.0.0.1:7860/gradio_api/mcp/"
SAMPLE_CSV = "data/state_tomo_data.csv"            # run this script in the project folder
EXPECTED_TOOLS = 32
results: list[tuple[bool, str]] = []


def check(ok: bool, what: str):
    results.append((bool(ok), what))
    print(f"[{'PASS' if ok else 'FAIL'}] {what}")


def close(a, b, tol=1e-4) -> bool:
    return a is not None and abs(float(a) - b) <= tol


class Client:
    def __init__(self, s: ClientSession, prefix: str):
        self.s, self.prefix, self.seconds = s, prefix, {}

    async def call(self, name: str, **args):
        """(JSON result or error text, list of image bytes, is_error)."""
        t = time.time()
        res = await self.s.call_tool(self.prefix + name, args)
        self.seconds[name] = max(self.seconds.get(name, 0), time.time() - t)
        texts = [c.text for c in res.content if c.type == "text" and not c.text.startswith("Image URL:")]
        images = [base64.b64decode(c.data) for c in res.content if c.type == "image"]
        if res.isError:
            return " ".join(texts), images, True
        return json.loads(texts[0]), images, False

    async def ok(self, name: str, **args) -> dict:
        out, _, err = await self.call(name, **args)
        if err:
            raise AssertionError(f"{name} failed: {out}")
        return out


async def main(url: str):
    async with http_client(url) as (r, w, _):
        async with ClientSession(r, w) as s:
            await s.initialize()
            tools = (await s.list_tools()).tools
            names = [t.name for t in tools]
            prefix = next((n[: -len("list_states")] for n in names if n.endswith("list_states")), "")
            c = Client(s, prefix)
            check(len(tools) == EXPECTED_TOOLS, f"{len(tools)} tools listed (expected {EXPECTED_TOOLS}); prefix '{prefix}'")
            bare = {n[len(prefix):] for n in names}
            check(not ({"respond", "new_session_id", "run", "lambda"} & bare),
                  "no chat, Workbench or helper endpoints exposed")
            check(all(t.description and len(t.description) > 60 for t in tools), "every tool has a description")
            undocumented = [f"{t.name}.{p}" for t in tools for p, v in t.inputSchema.get("properties", {}).items()
                            if not v.get("description")]
            check(not undocumented, f"every parameter has a description {undocumented[:3]}")

            # --- states and measures (closed forms)
            out = await c.ok("create_state", kind="phi+")
            sid = out["session_id"]
            check(sid.startswith("mcp_") and out["handle"] == "rho_1" and "session_note" in out,
                  "first call opens a session and returns session_id and handle")
            m = await c.ok("compute_measures", handle="rho_1", session_id=sid)
            check(close(m["concurrence"], 1) and close(m["negativity"], 0.5) and close(m["chsh_max"], 2 * math.sqrt(2))
                  and close(m["entanglement_of_formation"], 1) and close(m["purity"], 1),
                  "phi+: C = 1, N = 0.5, EoF = 1, CHSH = 2√2, purity 1")
            out = await c.ok("create_state", kind="werner", p=0.8, session_id=sid)
            m = await c.ok("compute_measures", handle=out["handle"], session_id=sid)
            check(close(m["concurrence"], 0.7) and close(m["chsh_max"], 2 * math.sqrt(2) * 0.8)
                  and close(m["purity"], (1 + 3 * 0.64) / 4), "Werner p=0.8: C = (3p-1)/2, CHSH = 2√2 p, purity (1+3p²)/4")
            t = await c.ok("teleportation_fidelity", handle=out["handle"], session_id=sid)
            check(close(t["fully_entangled_fraction"], 0.85) and close(t["teleportation_fidelity"], 0.9),
                  "Werner p=0.8: f = (1+3p)/4 = 0.85, F = (2f+1)/3 = 0.9")
            psi = (await c.ok("create_state", kind="psi-", session_id=sid))["handle"]
            h = (await c.ok("apply_global_noise", handle=psi, channel="collective_dephasing", param=0.5,
                            session_id=sid))["handle"]
            m = await c.ok("compute_measures", handle=h, session_id=sid)
            check(close(m["concurrence"], 1) and close(m["entanglement_of_formation"], 1),
                  "psi- is immune to collective dephasing: C = 1 and EoF = 1 (was NaN before the fix)")
            h = (await c.ok("apply_global_noise", handle="rho_1", channel="collective_dephasing", param=0.5,
                            session_id=sid))["handle"]
            m = await c.ok("compute_measures", handle=h, session_id=sid)
            check(close(m["concurrence"], 0.5 ** 4), "phi+ under collective dephasing p=0.5: C = (1-p)^4")
            p0 = 1 - 1 / math.sqrt(3)
            h = (await c.ok("apply_local_noise", handle="rho_1", channel="depolarizing", param=p0, session_id=sid))["handle"]
            m = await c.ok("compute_measures", handle=h, session_id=sid)
            check(close(m["concurrence"], 0, 1e-6), "local depolarizing on phi+: C = 0 at p = 1 - 1/√3")
            ev, imgs, _ = await c.call("evolve_state", handle="rho_1", process="dephasing", gamma=1.0, t_max=1.0,
                                       session_id=sid)
            m = await c.ok("compute_measures", handle=ev["final_state"], session_id=sid)
            check(close(m["concurrence"], math.exp(-2)) and len(imgs) == 1 and imgs[0][:4] == b"\x89PNG",
                  "Lindblad dephasing γ=1 on both photons: C(t=1) = e^(-2γt), plot returned as PNG")
            wp = await c.ok("apply_waveplate", handle="rho_1", plate="HWP", angle_deg=45.0, qubit="B", session_id=sid)
            psip = (await c.ok("create_state", kind="psi+", session_id=sid))["handle"]
            f = await c.ok("compare_states", handle_a=wp["new_handle"], handle_b=psip, session_id=sid)
            check(close(f["fidelity"], 1) and close(f["trace_distance"], 0), "HWP at 45° on photon B turns phi+ into psi+")
            q = await c.ok("qkd_key_rate", qber_z=0.05)
            hq = -0.05 * math.log2(0.05) - 0.95 * math.log2(0.95)
            check(close(q["secret_key_fraction"], 1 - 2 * hq, 1e-4), "BB84 key fraction 1 - 2h(0.05) (no session needed)")
            b = await c.ok("bit_flip_code", p=0.1)
            check(close(b["p_logical"], 3 * 0.01 - 2 * 0.001), "repetition code: p_L = 3p² - 2p³")
            k = await c.ok("get_kraus_operators", channel="amplitude_damping", param=0.3)
            check(k["number_of_operators"] == 2 and k["trace_preserving"], "amplitude-damping Kraus operators complete")

            # --- tomography: pasted CSV text, a simulated experiment, certification
            with open(SAMPLE_CSV, encoding="utf-8") as fh:          # the example data shipped with the app
                csv = fh.read()
            tomo = await c.ok("reconstruct_state_from_counts", csv_text=csv, error_bars=False, session_id=sid)
            ref = await c.ok("reconstruct_state_from_counts", csv_path=SAMPLE_CSV, error_bars=False, session_id=sid)
            m = await c.ok("compute_measures", handle=tomo["handle"], session_id=sid)
            m_ref = await c.ok("compute_measures", handle=ref["handle"], session_id=sid)
            check(tomo["data"]["file"].startswith("uploads/") and tomo["data"]["total_counts"] == ref["data"]["total_counts"]
                  and close(m["concurrence"], m_ref["concurrence"], 1e-9),
                  f"pasted CSV text = the same file read on the server: C = {m['concurrence']:.4f}, "
                  f"{tomo['data']['total_counts']} counts")
            sim = await c.ok("simulate_tomography", handle="rho_2", seed=1, session_id=sid)
            check(sim["csv_file"].startswith("uploads/") and "sessions" not in json.dumps(sim),
                  "simulated counts file is shown as uploads/..., no internal folder")
            again = await c.ok("reconstruct_state_from_counts", csv_path=sim["csv_file"], count_column="counts_raw",
                               error_bars=False, session_id=sid)
            check(again["key_numbers"]["concurrence"] == sim["reconstructed_raw"]["concurrence"],
                  "the simulated counts file, reconstructed again in the same session, gives the same concurrence")
            states = (await c.ok("list_states", session_id=sid))["states"]
            fitted = [x["handle"] for x in states if x["from_counts"]]
            cert = await c.ok("certify_correlations", handle=fitted[0], session_id=sid)
            check(any(x.get("holds") for x in cert["tests"]), "certification of a reconstructed state runs")
            fig, imgs, err = await c.call("plot_tomography_fit", handle=fitted[0], session_id=sid)
            check(not err and len(imgs) == 1, "tomography-fit diagnostics returned as an image")

            # --- plots
            for name, args in [("plot_density_matrix", {"handle": "rho_1", "style": "magnitude_phase"}),
                               ("plot_sweep", {"sweep": "werner_p", "quantities": ["concurrence", "chsh_max"]}),
                               ("plot_bloch_and_correlations", {"handles": ["rho_1", "rho_2"]}),
                               ("plot_fringes", {"handle": "rho_2"}),
                               ("plot_measures_vs_noise", {"start_kind": "phi+", "channel": "amplitude_damping"})]:
                out, imgs, err = await c.call(name, session_id=sid, **args)
                check(not err and len(imgs) == 1 and imgs[0][:4] == b"\x89PNG" and "figures/" not in json.dumps(out),
                      f"{name}: JSON + one PNG ({len(imgs[0]) // 1024 if imgs else 0} kB), no internal path")
            sw = await c.ok("plot_sweep", sweep="werner_p", quantities=["concurrence"], session_id=sid)
            cross = json.dumps(sw["results"])
            check("0.333" in cross, "Werner sweep: concurrence crosses 0 at p = 1/3")

            # --- sessions are separate; errors and limits
            other = await c.ok("create_state", kind="psi-")
            check(other["session_id"] != sid and other["handle"] == "rho_1", "a second client gets its own session")
            m = await c.ok("compute_measures", handle="rho_1", session_id=other["session_id"])
            check(close(m["concurrence"], 1) and (await c.ok("list_states", session_id=other["session_id"]))["states"][0]
                  ["description"] == "psi-", "same handle name, different session, different state")
            for what, name, args, needle in [
                    ("unknown handle", "compute_measures", {"handle": "rho_99", "session_id": sid}, "Unknown handle"),
                    ("made-up session id", "compute_measures", {"handle": "rho_1", "session_id": "mcp_" + "0" * 32},
                     "Unknown or expired"),
                    ("the command-line session", "list_states", {"session_id": "default"}, "Unknown or expired"),
                    ("a file outside data/ and uploads/", "reconstruct_state_from_counts",
                     {"csv_path": "../app.py", "session_id": sid}, "csv"),
                    ("the .env file", "reconstruct_state_from_counts", {"csv_path": ".env", "session_id": sid}, "csv"),
                    ("no session_id where states are needed", "compute_measures", {"handle": "rho_1"}, "session_id"),
                    ("another session's file", "reconstruct_state_from_counts",
                     {"csv_path": sim["csv_file"], "session_id": other["session_id"]}, "not found"),
                    ("another session's file by its folder", "reconstruct_state_from_counts",
                     {"csv_path": f"data/sessions/{sid}/" + sim["csv_file"].split("/")[-1],
                      "session_id": other["session_id"]}, "not found"),
                    ("too many sweep points", "plot_measures_vs_noise",
                     {"start_kind": "phi+", "channel": "depolarizing", "n_points": 100000, "session_id": sid}, "n_points")]:
                out, _, err = await c.call(name, **args)
                check(err and needle.lower() in str(out).lower(), f"refused: {what} ({str(out)[:70]}...)")

            # --- several clients at once (the compute lock serializes the work; nothing gets mixed up)
            async def one(kind: str, p: float):
                o = await c.ok("create_state", kind="werner", p=p, bell=kind)
                mm = await c.ok("compute_measures", handle=o["handle"], session_id=o["session_id"])
                return mm["concurrence"]
            got = await asyncio.gather(*(one(k, p) for k, p in [("phi+", 0.5), ("psi-", 0.7), ("phi-", 0.9)]))
            check(all(close(g, (3 * p - 1) / 2) for g, p in zip(got, (0.5, 0.7, 0.9))),
                  "three clients in parallel each get their own correct result")
            await c.ok("end_session", session_id=other["session_id"])
            out, _, err = await c.call("list_states", session_id=other["session_id"])
            check(err, "end_session deletes the session")

            slow = sorted(c.seconds.items(), key=lambda x: -x[1])[:3]
            print("slowest calls: " + ", ".join(f"{n} {t:.1f} s" for n, t in slow))
    bad = [w for ok, w in results if not ok]
    print(f"\n{len(results) - len(bad)} of {len(results)} checks passed.")
    return 1 if bad else 0



if __name__ == "__main__":
    sys.exit(asyncio.run(main(sys.argv[1] if len(sys.argv) > 1 else LOCAL)))
