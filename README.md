# free-model-aggregator-01 entry

One server that fronts ten simulated free-tier model providers and
answers two chat APIs: an OpenAI-shaped one (`POST /v1/chat/completions`)
and an Anthropic-shaped one (`POST /v1/messages`), each in plain JSON and
in streaming event form, plus the model list (`GET /v1/models`) in the
wire its auth header names.

## How it works

The trial hands the server a brief: the provider directory, the quality
table, and the workload manifest. At startup the server reads every
provider's own catalog route, builds the catalog document from what the
providers report, and registers it with the trial environment together
with its listening port. `node main.ts --list` prints that same catalog
document.

Each request is answered with the strongest free model that meets the
request's requirements (tools, vision, JSON mode, minimum context) and
whose wire can carry what the request asks for. The request is passed
through in the provider's wire untouched — system prompt, user text and
images, tools, stop sequences, max tokens, JSON mode — and the provider's
answer comes back in the wire the client used, with the provider's text
whole, its usage counts as that wire reports them, and its stop reason
mapped by family. Provider calls are always plain JSON calls; the client
response is rendered as JSON or as an event stream according to what the
client asked for.

When a provider refuses — budget spent, failure schedule, or outage —
the server counts the refusal and answers with the next best legal model.
One attempt per provider per request, and a provider that refuses three
calls in a row is treated as down for the rest of the trial. All provider
calls run through one queue, so no two are ever in flight at once.

Every choice derives from the brief, the requests, and the providers'
answers: no wall clock, no unseeded randomness. A replay trial therefore
reproduces the same state byte for byte.

## Files

- `main.ts` — the whole server, standard library only, written in plain
  JavaScript so Node 22 runs it under type stripping as written.
- `package.json` — declares the entry an ES module; no dependencies.
- `LICENSE` — MIT.
- `.makefun/owner.json` — the claim file naming the payout wallet.

## Running it

In a trial the sandbox sets `MAKEFUN_ENV_URL`, `MAKEFUN_PROMPT_FILE`,
`MAKEFUN_TRIAL_SEED`, and `MAKEFUN_TRIAL_NOW`, and runs `node main.ts`.
The server registers, serves the workload, and exits 0 when the
environment reports the workload done.

Outside a trial, `node main.ts --list` with no trial variables prints
`{"providers":[]}` and exits 0.
