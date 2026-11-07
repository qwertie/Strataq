# Engine requests and command replies

An engine advertises `requests` on its `READY` line. Python then uses this
protocol for every generation, including when only one request is running.
There is no solo request, client-selected execution slot, admission handoff,
or promotion between generation modes.

```text
RUN <request-id> <max-new> [key=value ...] <token,token,...>
RUNI <request-id> <max-new> [key=value ...] <picture-file> <token,token,...>
CANCEL <request-id>
ASK <command-id> <command>
```

IDs contain letters, digits, underscores and hyphens. Python assigns IDs per
engine process. A generation keeps its identity if it resumes after preemption.
Sampling options and picture files have the same meanings as before.

Generation events are complete lines prefixed with `R <request-id>`:

```text
R r1 STARTED
R r1 PP <read> <total> <milliseconds> <tokens-per-second>
R r1 T <token-id>
R r1 DONE <produced> <prompt-tokens> <prompt-ms> <decode-ms> <finish> [statistics ...]
R r1 ERR <reason>
```

`STARTED` means the engine has admitted the request. Queue deadlines remain
active until then. `DONE` supplies the engine's existing statistics, including
its optional model-specific tail. A `preempted` completion preserves output:
Python resubmits the same identity with the accepted output appended to its
prompt and the remaining output allowance. It combines the segment statistics.
Cancellation targets that identity, including while it waits for admission.
An unknown or already-completed cancellation has no effect.

Every line of an ASK reply is prefixed with `C <command-id>`. The command's
existing end marker still ends the reply:

```text
ASK c1 CHANGE_SETTINGS rows=4
C c1 MEMORY ...
C c1 SETTINGS rows=4 ...
C c1 END_MEMORY
```

Python routes each identity to its own queue. A generation does not hold a
command lock. A timed-out command's late reply is discarded, so it cannot be
mistaken for a later command's answer. Process exit wakes all queues; requests
from a previous process cannot write into its successor.

The paged engine chooses execution placement and admission priority. Its
`request_capacity` is a scheduling capacity, not a set of empty conversations.
`request_queue=1` tells Python to let the engine queue and preempt requests.
Conversation state and working buffers are created on demand. No extra state
is reserved for a distinguished request. Cached contexts have independent
`conversation<N>` addresses for inspection.

The paged command loop runs between decode steps, so settings can change while
one or several requests are generating. Commands still wait for a model
operation that is in progress. Prompt admission currently completes its prefill
before returning to that loop. Command identities keep late replies safe if a
long prefill outlasts a caller's timeout.

The older Qwen3.8, Qwen3.6, DeepSeek and SYCL handlers use the shared transport
adapter. It chooses their internal placement, frames their events, and combines
their admission and completion statistics into one DONE. Their `GEN`, `BGEN`,
`BADM`, `BT` and `BDONE` details stay inside the adapter. Sequential engines can
still finish an operation before servicing another command; this protocol does
not make an old model implementation concurrent or add live settings support.

Python retains its old protocol client for installed engines that do not
advertise `requests`. It never chooses that client for an updated engine.

Build `request_protocol_driver` and `request_test` to exercise the shared C++
transport without loading a model.
Run `python -m pytest strata_server/test_request_protocol.py`. Set `STRATA_REQUEST_PROTOCOL_DRIVER` to
the executable's full path when the build is outside `build` or `build-hip-win`.
These checks cover event routing, settings during generation, lazy startup,
cancellation, preemption continuation, admission deadlines and late replies.
They do not establish model-backed GPU correctness.
