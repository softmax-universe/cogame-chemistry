## The decision layer: tolerant parsing, the reply schema, and the guarantee
## that a broken transport never raises and never blocks the episode.

import std/[json, os, strutils, unicode, unittest]
import chemistry/sim
import chemistry/llm

proc freshSim(cycles = 3, distractorPeriod = 2): Sim =
  var config = defaultGameConfig()
  config.seed = 5
  config.cycles = cycles
  config.distractorPeriod = distractorPeriod
  config.distractorGroundCap = 24
  config.llmTimeoutSeconds = 5
  result = initSim(config)
  result.logEnabled = false
  ## The server marks a seat connected when its player socket upgrades; the
  ## decision layer only calls the LLM for connected seats.
  for slot in 0 ..< Seats:
    result.cogs[slot].connected = true

suite "extractJsonObject is strict":
  test "a raw JSON object":
    let node = extractJsonObject("{\"job\":\"idle\"}")
    check node{"job"}.getStr() == "idle"

  test "a markdown-fenced reply":
    let node = extractJsonObject("```json\n{\"job\":\"idle\"}\n```")
    check node{"job"}.getStr() == "idle"

  test "a prose-prefixed reply with trailing prose is rejected":
    expect ChemistryError:
      discard extractJsonObject(
        "Sure! Here is my order:\n{\"job\": \"forage\"}\nHope that helps.")

  test "a reply with no object at all raises":
    expect ChemistryError:
      discard extractJsonObject("I would rather not say.")

suite "the reply schema":
  let sim = freshSim()

  test "an unknown job is invalid":
    expect ChemistryError:
      discard sim.parseDecision(parseJson("""{"job":"sabotage"}"""))

  test "a missing job is invalid":
    expect ChemistryError:
      discard sim.parseDecision(parseJson("""{"say":"hello"}"""))

  test "supply without a reactor is invalid":
    expect ChemistryError:
      discard sim.parseDecision(
        parseJson("""{"job":"supply","molecule":"resin"}"""))

  test "supply without a molecule is invalid":
    expect ChemistryError:
      discard sim.parseDecision(
        parseJson("""{"job":"supply","reactor":"amber"}"""))

  test "a species absent from this variant is invalid":
    let plain = freshSim(3, 0)
    expect ChemistryError:
      discard plain.parseDecision(parseJson(
        """{"job":"supply","molecule":"glitter","reactor":"amber"}"""))

  test "a reactor absent from this variant is REJECTED, not clamped":
    var two = freshSim(2, 0)
    two.reactors[1].charge = 0
    expect ChemistryError:
      discard two.parseDecision(parseJson(
        """{"job":"supply","molecule":"spark","reactor":"cobalt"}"""))

  test "a reactor absent from this variant is REJECTED on forage too":
    var two = freshSim(2, 0)
    two.reactors[1].charge = 0
    expect ChemistryError:
      discard two.parseDecision(parseJson(
        """{"job":"forage","reactor":"cobalt"}"""))

  test "non-string field values are rejected without coercion":
    expect ChemistryError:
      discard sim.parseDecision(parseJson("""{"job":"idle","molecule":123}"""))
    expect ChemistryError:
      discard sim.parseDecision(parseJson("""{"job":"idle","reactor":true}"""))
    expect ChemistryError:
      discard sim.parseDecision(parseJson("""{"job":"idle","say":42}"""))

  test "a feedstock the named reactor does not take is ACCEPTED as written":
    ## The misdrop is the graph test and must stay expressible.
    let order = sim.parseDecision(parseJson(
      """{"job":"supply","molecule":"brine","reactor":"amber"}"""))
    check order.job == jobSupply
    check order.molecule == spBrine
    check order.reactor == rxAmber
    check not order.clamped

  test "say and notes are cut on rune boundaries":
    let long = "\u00e9".repeat(500)
    let order = sim.parseDecision(%*{
      "job": "idle", "say": long, "notes": long})
    check order.say.len <= MaxSayLen * 4
    check order.notes.len <= MaxNotesLen * 4
    check validateUtf8(order.say) == -1
    check validateUtf8(order.notes) == -1

suite "no fallbacks: reject and record missed turns":
  test "with no credentials unscripted seats miss their turn without fallback":
    delEnv("COWORLD_LLM_ENDPOINT")
    delEnv("AWS_ENDPOINT_URL_BEDROCK_RUNTIME")
    delEnv("AWS_BEARER_TOKEN_BEDROCK")
    delEnv("ANTHROPIC_API_KEY")
    delEnv("ANTHROPIC_API_KEY_URI")
    let sim = freshSim()
    let client = newLlmClient(sim.config)
    check client.disabled
    var prompts = newSeq[string](Seats)
    var scripted = newSeq[ScriptKind](Seats)
    let decisions = client.decideAll(sim, prompts, scripted)
    check decisions.len == Seats
    for slot in 0 ..< Seats:
      check not decisions[slot].hasOrder
      check decisions[slot].status == asMissing

  test "a dead transport rejects and marks missed turns without substituting an order":
    ## A closed port on localhost is a transport that fails immediately: the
    ## same path a timeout, a 429, a 403 or a junk body takes.
    putEnv("AWS_ENDPOINT_URL_BEDROCK_RUNTIME", "http://127.0.0.1:9/bedrock")
    putEnv("AWS_BEARER_TOKEN_BEDROCK", "not-a-real-token")
    defer:
      delEnv("AWS_ENDPOINT_URL_BEDROCK_RUNTIME")
      delEnv("AWS_BEARER_TOKEN_BEDROCK")
    let sim = freshSim()
    let client = newLlmClient(sim.config)
    check not client.disabled
    var prompts = newSeq[string](Seats)
    for slot in 0 ..< Seats:
      prompts[slot] = "hold your lane"
    var scripted = newSeq[ScriptKind](Seats)
    let decisions = client.decideAll(sim, prompts, scripted)
    ## ONE batch carries every open seat -- eight on shift 1.
    check client.lastBatchSize == Seats
    for slot in 0 ..< Seats:
      check not decisions[slot].hasOrder
      check decisions[slot].status == asRejected
      check decisions[slot].error.len > 0

  test "scripted seats never enter the batch":
    let sim = freshSim()
    delEnv("AWS_ENDPOINT_URL_BEDROCK_RUNTIME")
    delEnv("AWS_BEARER_TOKEN_BEDROCK")
    putEnv("ANTHROPIC_API_KEY", "sk-test-not-real")
    defer: delEnv("ANTHROPIC_API_KEY")
    let client = newLlmClient(sim.config)
    check not client.disabled
    var prompts = newSeq[string](Seats)
    var scripted = newSeq[ScriptKind](Seats)
    for slot in 0 ..< Seats:
      scripted[slot] = (if slot mod 2 == 0: skCourier else: skFreeloader)
    let decisions = client.decideAll(sim, prompts, scripted)
    check client.lastBatchSize == 0
    for slot in 0 ..< Seats:
      check decisions[slot].hasOrder
      check decisions[slot].order.source == osScripted

  test "a disconnected seat misses its turn and never enters the batch":
    putEnv("AWS_ENDPOINT_URL_BEDROCK_RUNTIME", "http://127.0.0.1:9/bedrock")
    putEnv("AWS_BEARER_TOKEN_BEDROCK", "not-a-real-token")
    defer:
      delEnv("AWS_ENDPOINT_URL_BEDROCK_RUNTIME")
      delEnv("AWS_BEARER_TOKEN_BEDROCK")
    var sim = freshSim()
    sim.cogs[2].connected = false
    sim.cogs[5].connected = false
    let client = newLlmClient(sim.config)
    check not client.disabled
    var prompts = newSeq[string](Seats)
    var scripted = newSeq[ScriptKind](Seats)
    let decisions = client.decideAll(sim, prompts, scripted)
    check client.lastBatchSize == Seats - 2
    for slot in [2, 5]:
      check not decisions[slot].hasOrder
      check decisions[slot].status == asMissing
    for slot in [0, 1, 3, 4, 6, 7]:
      check not decisions[slot].hasOrder
      check decisions[slot].status == asRejected

suite "the observation and the prompts":
  let sim = freshSim()

  test "the observation carries the room but never another seat's secrets":
    let node = sim.observationJson(3)
    check node{"protocol"}.getStr() == "chemistry.player.v2"
    check node{"slot"}.getInt() == 3
    check node{"name"}.getStr() == SeatAliases[3]
    check node{"cogs"}.len == Seats
    check node{"reactors"}.len == sim.reactors.len
    check node{"rules"}{"reactions"}.len == sim.reactors.len
    let text = $node
    ## Aliases in-game; no policy name, player name or account ever reaches a
    ## seat.
    for alias in SeatAliases:
      check alias in text
    check "chemistry-foreman" notin text
    check "daveey" notin text
    ## Only the seat's OWN notes are present.
    check node{"notes"}.kind == JString

  test "the prompts name the legal enum values for THIS variant":
    let two = freshSim(2, 0)
    let user = two.userPrompt(0, "operator guidance here")
    check "operator guidance here" in user
    check "cobalt" notin user
    check "glitter" notin user
    let system = two.systemPrompt(0)
    check "ARGON" in system
    check "Two vats (reactors)" in system
    check "glitter" notin system
    check "cobalt" notin system
    check "Example:" in system
    check "reply with ONLY one JSON object" in system
    check "supply | forage | hoard | idle" in system
