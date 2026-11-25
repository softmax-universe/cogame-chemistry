## Claude-backed decision making for Chemistry.
##
## Forked from `cogame-bullwhip/src/bullwhip/llm.nim` (the game-side batched
## decision layer paintbot does not have). A policy is just a prompt: the game
## composes each seat's observation plus that seat's `PLAYER_PROMPT` and asks
## Claude for ONE standing order for the next shift.
##
## Decisions within a shift are simultaneous by rule, so all eight seats'
## requests go out as ONE parallel batch (`curly.makeRequests`); invalid
## replies are retried once as a smaller batch carrying a hint, and anything
## still failing is rejected without executing an order on that seat's behalf.
##
## Credentials, in order of preference:
##   COWORLD_LLM_ENDPOINT            - hosted sidecar
##   Bedrock bearer token            - local play
##   ANTHROPIC_API_KEY                - the key itself
##   ANTHROPIC_API_KEY_URI            - a URI holding the key
## With none the client disables itself immediately and unscripted model seats
## miss their turns; explicitly scripted seats still run their declared policy.

import
  std/[json, options, os, strutils, times],
  bitworld/runtime,
  curly,
  sim,
  prompts, training

export prompts, ActionStatus

const
  AnthropicUrl = "https://api.anthropic.com/v1/messages"
  AnthropicVersion = "2023-06-01"
  BedrockAnthropicVersion = "bedrock-2023-05-31"

type
  ChemistryInfraError* = object of ChemistryError

  LlmTransport = enum
    ltNone, ltSidecar, ltBedrock, ltAnthropic

  LlmClient* = ref object
    curl: Curly
    transport: LlmTransport
    apiKey: string
    sidecarEndpoint: string
    bedrockEndpoint: string
    bedrockModels: seq[string]
    bedrockModel: int
    bedrockToken: string
    model: string
    maxOutputTokens: int
    timeoutSeconds: int
    disabled*: bool
    nativeAttempts*: seq[seq[DecisionAttempt]]
    lastBatchSize*: int   ## seats carried by the most recent request batch

  SeatDecision* = object
    hasOrder*: bool
    order*: Order
    status*: ActionStatus
    error*: string
    infraFailure*: bool

proc resolveApiKey(): string =
  result = getEnv("ANTHROPIC_API_KEY").strip()
  if result.len > 0:
    return
  let uri = getEnv("ANTHROPIC_API_KEY_URI").strip()
  if uri.len == 0:
    return ""
  try:
    result = readCogameUri(uri, "ANTHROPIC_API_KEY_URI").strip()
  except CatchableError as error:
    echo "chemistry llm: failed to fetch ANTHROPIC_API_KEY_URI: ", error.msg
    result = ""

proc bedrockModelIds*(): seq[string] =
  ## Haiku ONLY. The sonnet inference profiles time out on every sidecar call
  ## and turn one throttle into a cascade of scripted fallbacks (raid,
  ## 2026-08-23), so this ladder has exactly one rung. BEDROCK_MODEL pins a
  ## different id when an operator needs one.
  let pinned = getEnv("BEDROCK_MODEL").strip()
  if pinned.len > 0:
    return @[pinned]
  @["us.anthropic.claude-haiku-4-5-20251001-v1:0"]

proc tryNextBedrockModel(client: LlmClient, why: string): bool =
  if client.transport != ltBedrock or
      client.bedrockModel + 1 >= client.bedrockModels.len:
    return false
  client.bedrockModel.inc
  echo "chemistry llm: ", client.bedrockModels[client.bedrockModel - 1],
    " unusable (", why, "); falling back to ",
    client.bedrockModels[client.bedrockModel]
  true

proc bedrockUrl(client: LlmClient): string =
  client.bedrockEndpoint & "/model/" &
    client.bedrockModels[client.bedrockModel] & "/invoke"

proc newLlmClient*(config: GameConfig): LlmClient =
  result = LlmClient(
    model: config.model,
    maxOutputTokens: config.maxOutputTokens,
    timeoutSeconds: config.llmTimeoutSeconds
  )
  let sidecarEndpoint = getEnv("COWORLD_LLM_ENDPOINT").strip()
  if sidecarEndpoint.len > 0:
    result.transport = ltSidecar
    result.sidecarEndpoint = sidecarEndpoint.strip(chars = {'/'}, leading = false)
    result.model = getEnv("COWORLD_LLM_MODEL", "anthropic/claude-haiku-4.5")
    result.curl = newCurly()
    return
  let bedrockEndpoint = getEnv("AWS_ENDPOINT_URL_BEDROCK_RUNTIME").strip()
  let bedrockToken = getEnv("AWS_BEARER_TOKEN_BEDROCK").strip()
  if bedrockEndpoint.len > 0 or bedrockToken.len > 0:
    let region = getEnv("AWS_REGION", getEnv("AWS_DEFAULT_REGION", "us-west-2"))
    let endpoint =
      if bedrockEndpoint.len > 0: bedrockEndpoint
      else: "https://bedrock-runtime." & region & ".amazonaws.com"
    result.transport = ltBedrock
    result.bedrockEndpoint = endpoint.strip(chars = {'/'}, leading = false)
    result.bedrockModels = bedrockModelIds()
    result.bedrockToken = bedrockToken
    result.curl = newCurly()
    echo "chemistry llm: bedrock transport, model ",
      result.bedrockModels[result.bedrockModel], ", url ", result.bedrockUrl
    return
  result.apiKey = resolveApiKey()
  if result.apiKey.len > 0:
    result.transport = ltAnthropic
    result.curl = newCurly()
    echo "chemistry llm: anthropic transport, model ", result.model
  else:
    result.transport = ltNone
    result.disabled = true
    echo "chemistry llm: no LLM credentials; unscripted model seats will miss turns"

# ---- The observation ---------------------------------------------------------

proc reactorJson(sim: Sim, index: int): JsonNode =
  let reactor = sim.reactors[index]
  var feeds = newJArray()
  for species in reactor.feed:
    feeds.add(%($species))
  %*{
    "name": $reactor.name,
    "cell": [reactor.cell.x, reactor.cell.y],
    "feedstocks": feeds,
    "charge": reactor.charge,
    "chargeMax": sim.config.chargeMax,
    "status": $sim.reactorStatus(index),
    "yieldNow": sim.yieldNow(index),
    "stock": {$reactor.feed[0]: reactor.stock[0],
              $reactor.feed[1]: reactor.stock[1]},
    "cooldown": reactor.cooldown,
    "ticksSinceReaction": reactor.ticksSinceReaction,
    "foodMade": reactor.foodMade,
    "coldStartCost": sim.config.coldStartCost
  }

proc nearestCellOf(sim: Sim, slot: int, species: Species): JsonNode =
  let cells = sim.looseCells(species)
  if cells.len == 0:
    return newJNull()
  let field = sim.room.distanceField(cells)
  var best = Cell(x: -1, y: -1)
  var bestDistance = high(int)
  for cell in cells:
    let distance = manhattan(cell, sim.cogs[slot].cell)
    if distance < bestDistance:
      bestDistance = distance
      best = cell
  discard field
  %*[best.x, best.y]

proc orderJson(order: Order, withSource: bool): JsonNode =
  result = %*{"job": $order.job}
  if order.hasMolecule:
    result["molecule"] = %($order.molecule)
  if order.hasReactor:
    result["reactor"] = %($order.reactor)
  if withSource:
    result["source"] = %($order.source)

proc legalOrders*(sim: Sim): JsonNode =
  ## Exact standing orders available to any external policy in this variant.
  result = newJArray()
  result.add(%*{"id": "idle", "job": "idle"})
  result.add(%*{"id": "forage", "job": "forage"})
  for reactor in sim.reactors:
    result.add(%*{"id": "forage:" & $reactor.name,
      "job": "forage", "reactor": $reactor.name})
  for species in sim.config.speciesPresent():
    result.add(%*{"id": "hoard:" & $species,
      "job": "hoard", "molecule": $species})
    for reactor in sim.reactors:
      result.add(%*{"id": "supply:" & $species & ":" & $reactor.name,
        "job": "supply", "molecule": $species,
        "reactor": $reactor.name})

proc observationJson*(sim: Sim, slot: int): JsonNode =
  ## The `state` frame each seat gets at every shift boundary. Everything in
  ## here is visible to that seat; NOTHING else is -- in particular no policy
  ## name, no other seat's notes and no other seat's order for the shift about
  ## to be played (decisions are simultaneous).
  let cog = sim.cogs[slot]
  var reactors = newJArray()
  for index in 0 ..< sim.reactors.len:
    reactors.add sim.reactorJson(index)
  var molecules = newJObject()
  for species in sim.config.speciesPresent():
    molecules[$species] = %*{
      "inert": species.isInert(),
      "vent": [VentCells[species].x, VentCells[species].y],
      "loose": sim.looseCount(species),
      "nearestToYou": sim.nearestCellOf(slot, species)
    }
  var foodCellsJson = newJArray()
  for cell in sim.foodCells():
    foodCellsJson.add(%*[cell.x, cell.y])
  var cogs = newJArray()
  for other in 0 ..< Seats:
    let entry = sim.cogs[other]
    cogs.add(%*{
      "alias": entry.alias,
      "cell": [entry.cell.x, entry.cell.y],
      "carrying": (if entry.hasCarry: %($entry.carrying) else: newJNull()),
      "foodEaten": entry.foodEaten,
      "delivered": entry.delivered,
      "misdrops": entry.misdrops,
      "hoard": entry.hoard,
      "lastOrder": (if entry.hasOrder: orderJson(entry.order, false) else: newJNull()),
      "missedTurn": entry.missedTurn,
      "missedTurns": entry.missedTurns,
      "say": entry.say
    })
  var history = newJArray()
  for record in sim.history:
    var eaten = newJArray()
    for slotEaten in record.eaten:
      eaten.add(%slotEaten)
    var reactions = newJArray()
    for value in record.reactions:
      reactions.add(%value)
    var foodMade = newJArray()
    for value in record.foodMade:
      foodMade.add(%value)
    var charges = newJArray()
    for value in record.charge:
      charges.add(%value)
    history.add(%*{
      "shift": record.shift,
      "reactions": reactions,
      "foodMade": foodMade,
      "eaten": eaten,
      "coldStarts": record.coldStarts,
      "misdrops": record.misdrops,
      "charge": charges
    })
  var rules = newJArray()
  for index in 0 ..< sim.reactors.len:
    let reactor = sim.reactors[index]
    rules.add(%*{
      "reactor": $reactor.name,
      "inputs": [$reactor.feed[0], $reactor.feed[1]],
      "output": "food",
      "yield": "1 + charge div 3",
      "requires": "charge >= 1 and both stocks >= 1"
    })
  var inert = newJArray()
  for species in sim.config.speciesPresent():
    if species.isInert():
      inert.add(%($species))
  %*{
    "type": "state",
    "protocol": "chemistry.player.v2",
    "slot": slot,
    "name": cog.alias,
    "shift": sim.shift + 1,
    "shifts": sim.config.shifts,
    "ticksPerShift": sim.config.ticksPerShift,
    "tick": sim.tick,
    "room": {"cols": RoomCols, "rows": RoomRows,
             "variant": sim.config.variantId()},
    "you": {
      "cell": [cog.cell.x, cog.cell.y],
      "carrying": (if cog.hasCarry: %($cog.carrying) else: newJNull()),
      "home": [cog.home.x, cog.home.y],
      "foodEaten": cog.foodEaten,
      "delivered": cog.delivered,
      "misdrops": cog.misdrops,
      "hoard": cog.hoard,
      "lastOrder": (if cog.hasOrder: orderJson(cog.order, true) else: newJNull()),
      "missedTurn": cog.missedTurn,
      "missedTurns": cog.missedTurns
    },
    "reactors": reactors,
    "molecules": molecules,
    "food": {"loose": sim.foodCells().len, "cells": foodCellsJson},
    "cogs": cogs,
    "history": history,
    "notes": cog.notes,
    "legalOrders": sim.legalOrders(),
    "rules": {
      "reactions": rules,
      "autocatalysis": "every reaction adds 1 charge (max " &
        $sim.config.chargeMax & "); yield rises with charge",
      "chargeDecay": "-1 charge per reactor every " &
        $sim.config.chargeDecayPeriod & " ticks (once per shift)",
      "coldStart": "at charge 0 a reactor consumes " &
        $sim.config.coldStartCost & " of EACH feedstock to return to " &
        "charge 1 and makes no food",
      "misdrop": "a molecule dropped on a reactor that does not take it " &
        "is absorbed and destroyed",
      "inert": inert,
      "carryCap": sim.config.carryCap,
      "moveCooldown": sim.config.moveCooldown,
      "foodLifetime": sim.config.foodLifetime,
      "ventPeriod": sim.config.ventPeriod,
      "scoring": "your score is the number of food tokens YOU eat; food is " &
        "eaten automatically by standing on it"
    }
  }

# ---- Prompts ----------------------------------------------------------------

proc titleCase(text: string): string =
  if text.len == 0: text else: text[0..0].toUpperAscii() & text[1 .. ^1]

proc legalJobs(): string = "supply | forage | hoard | idle"

proc legalMolecules(sim: Sim): string =
  var parts: seq[string]
  for species in sim.config.speciesPresent():
    parts.add $species
  parts.join(" | ")

proc legalReactors(sim: Sim): string =
  var parts: seq[string]
  for reactor in sim.reactors:
    parts.add $reactor.name
  parts.join(" | ")

proc inertSpeciesList(sim: Sim): string =
  var parts: seq[string]
  for species in sim.config.speciesPresent():
    if species.isInert():
      parts.add $species
  if parts.len == 0: "none" else: parts.join(", ")

proc graphTable(sim: Sim): string =
  let inert = sim.inertSpeciesList()
  var lines = @["reactor | takes | inert species never accepted"]
  for reactor in sim.reactors:
    lines.add($reactor.name & " | " & $reactor.feed[0] & " + " &
      $reactor.feed[1] & " | " & inert)
  lines.join("\n")

proc systemPrompt*(sim: Sim, slot: int): string =
  let cog = sim.cogs[slot]
  let vatCountWord =
    if sim.reactors.len == 2: "Two"
    elif sim.reactors.len == 3: "Three"
    else: $sim.reactors.len
  let introduction = "You are " & cog.alias.toUpperAscii() &
    ", one of eight cogs working a chemistry floor.\n"
  let room = "THE ROOM: a fixed " & $RoomCols & "x" & $RoomRows &
    " grid. Walls ring the edge and two 2x2 pillars sit in\n" &
    "the middle. " & vatCountWord & " vats (reactors) sit on 3x3 pads. Vents emit loose molecules\n" &
    "onto the floor. Cells are (col, row) with the origin at the top left.\n\n"
  let control = "WHAT YOU CONTROL: you do NOT emit per-tick moves. Once per SHIFT (" &
    $sim.config.ticksPerShift & " ticks)\n" &
    "you choose ONE standing order, and a deterministic courier kernel walks it for\n" &
    "you: it paths, picks up, carries and drops. Your per-tick action vocabulary --\n" &
    "move_n, move_s, move_e, move_w, take, drop, wait -- is what the kernel emits\n" &
    "on your behalf. You carry AT MOST " & $sim.config.carryCap &
    " molecule (carryCap " & $sim.config.carryCap & ") and may move once\n" &
    "every " & $sim.config.moveCooldown & " ticks.\n\n"
  let jobs = """THE JOBS:
- supply: fetch the named molecule and drop it into the named vat's stock.
- forage: walk to loose food, or camp the named vat's spill ring and wait (reactor is optional).
- hoard: fetch the named molecule and stack it on your own home cell. Hoarding
  scores NOTHING, ever.
- idle: stand still.

"""
  let graph = """THE REACTION GRAPH:
""" & sim.graphTable() & "\n"
  let coldStartTotal = sim.config.coldStartCost * 2
  let inertLine =
    if sim.config.distractorPeriod > 0:
      "- INERT species (" & sim.inertSpeciesList() &
        ") do nothing at all. They can be carried and hoarded, and they\n  are a trap.\n\n"
    else:
      "- No inert distractor species exist in this variant.\n\n"
  let mechanics = "\n" &
    "- A vat reacts when its charge is at least 1, its cooldown is 0 and BOTH its\n" &
    "  feedstocks are in stock: it consumes 1 of each, gains 1 charge (max " &
      $sim.config.chargeMax & ") and drops 1 + charge div 3 FOOD tokens.\n" &
    "- AUTOCATALYSIS: charge is the catalyst. Every reaction makes the next one\n" &
    "  more productive, but charge decays by 1 every " &
      $sim.config.chargeDecayPeriod & " ticks -- so\n" &
    "  \"running\" is a state you HOLD, not one you reach.\n" &
    "- COLD START: a vat at charge 0 cannot react. It returns to charge 1 only by\n" &
    "  consuming " & $sim.config.coldStartCost & " of EACH feedstock, and makes\n" &
    "  no food doing it. Letting a cycle die costs the room " & $coldStartTotal &
      " deliveries of pure\n  investment.\n" &
    "- MISDROP: a molecule dropped on a vat that does not take it -- an inert\n" &
    "  species, or the wrong feedstock -- is ABSORBED AND DESTROYED. Reading the\n" &
    "  graph is what stops you paying that.\n" & inertLine
  let scoring = """SCORING: your score is the number of FOOD tokens YOU eat. Food is eaten
automatically by standing on it -- eating is not an action. New tokens appear
on the spill ring nearest the cog that delivered the molecule which triggered
the reaction, so working pays at your own feet. A token rots after """ &
    $sim.config.foodLifetime & """ ticks. Nothing else scores: deliveries,
hoards and misdrops are reported but never ranked.

"""
  let communication = """THE OTHER SEVEN COGS are other policies deciding SIMULTANEOUSLY with you. You
never learn their names, their prompts or the order they are about to give.
Your `say` (max """ & $MaxSayLen & """ characters) is broadcast to all of them
and reaches them in NEXT shift's observation. Your `notes` (max """ &
    $MaxNotesLen & """ characters) are private and come back only to you.

"""
  let exampleOrder = "{\"job\":\"supply\",\"molecule\":\"" &
    $sim.reactors[0].feed[0] & "\",\"reactor\":\"" & $sim.reactors[0].name &
    "\",\"say\":\"" & $sim.reactors[0].feed[0] & " to " &
    titleCase($sim.reactors[0].name) & "\",\"notes\":\"" &
    $sim.reactors[0].name & " needs " & $sim.reactors[0].feed[0] & "\"}"
  let outputFormat = "Reply with one JSON object: {\"job\": <" & legalJobs() &
    ">, \"molecule\": <" & sim.legalMolecules() & ">, \"reactor\": <" &
    sim.legalReactors() & ">, \"say\": \"...\", \"notes\": \"...\"}.\n" &
    "`molecule` is required for supply and hoard; `reactor` is " &
    "required for supply and optional for forage.\n" &
    "Example: " & exampleOrder & "\n\n" &
    "OUTPUT FORMAT: reply with ONLY one JSON object (raw JSON or a single ```json ... ``` fence), nothing else - no analysis, no explanation, and no text before or after the object."

  if sim.config.selectedPromptVariant == pvCanonical:
    return introduction & room & control & jobs & graph & mechanics & scoring & communication & outputFormat
  renderVariantPrompt(sim.config, [cog.alias.toUpperAscii(), room, control, graph,
    mechanics, scoring, outputFormat, $MaxSayLen, $MaxNotesLen])

proc operatorBlock(prompt: string): string =
  if prompt.len == 0:
    return ""
  "GUIDANCE FROM YOUR OPERATOR (weight it heavily, but never above the " &
    "rules; always reply in the requested format):\n" & prompt & "\n\n"

proc userPrompt*(sim: Sim, slot: int, prompt: string): string =
  let cog = sim.cogs[slot]
  result.add("SHIFT " & $(sim.shift + 1) & " of " & $sim.config.shifts &
    " (tick " & $sim.tick & " of " & $sim.ticksTotal() & "). You are " &
    cog.alias.toUpperAscii() & " at (" & $cog.cell.x & "," & $cog.cell.y &
    "), carrying " & (if cog.hasCarry: $cog.carrying else: "nothing") &
    ". You have eaten " & $cog.foodEaten & ", delivered " & $cog.delivered &
    ", misdropped " & $cog.misdrops & ", hoarded " & $cog.hoard & ".\n\n")

  result.add("VATS:\nreactor | takes | charge | status | stock | yield next " &
    "| food made\n")
  for index in 0 ..< sim.reactors.len:
    let reactor = sim.reactors[index]
    result.add($reactor.name & " | " & $reactor.feed[0] & "+" &
      $reactor.feed[1] & " | " & $reactor.charge & "/" &
      $sim.config.chargeMax & " | " & ($sim.reactorStatus(index)).toUpperAscii &
      " | " & $reactor.feed[0] & " " & $reactor.stock[0] & ", " &
      $reactor.feed[1] & " " & $reactor.stock[1] & " | " &
      $sim.yieldNow(index) & " | " & $reactor.foodMade & "\n")

  result.add("\nMOLECULES:\nspecies | inert | loose | nearest to you\n")
  for species in sim.config.speciesPresent():
    let cells = sim.looseCells(species)
    var nearest = "none"
    var bestDistance = high(int)
    for cell in cells:
      let distance = manhattan(cell, cog.cell)
      if distance < bestDistance:
        bestDistance = distance
        nearest = "(" & $cell.x & "," & $cell.y & ")"
    result.add($species & " | " & (if species.isInert(): "YES" else: "no") &
      " | " & $cells.len & " | " & nearest & "\n")

  result.add("\nFOOD ON THE FLOOR: " & $sim.foodCells().len & "\n")

  result.add("\nCOGS:\nalias | cell | carrying | ate | delivered | hoard | " &
    "last job | last say\n")
  for other in 0 ..< Seats:
    let entry = sim.cogs[other]
    let jobText =
      if entry.missedTurn: "MISSED"
      elif not entry.hasOrder: "-"
      else:
        $entry.order.job &
          (if entry.order.hasMolecule: " " & $entry.order.molecule else: "") &
          (if entry.order.hasReactor: " -> " & $entry.order.reactor else: "")
    result.add(entry.alias & " | (" & $entry.cell.x & "," & $entry.cell.y &
      ") | " & (if entry.hasCarry: $entry.carrying else: "-") & " | " &
      $entry.foodEaten & " | " & $entry.delivered & " | " & $entry.hoard &
      " | " & jobText &
      " | " & (if entry.say.len > 0: "\"" & entry.say & "\"" else: "-") & "\n")

  if sim.history.len > 0:
    result.add("\nSHIFT HISTORY:\nshift | reactions | food made | you ate | " &
      "cold starts | misdrops | charges\n")
    for record in sim.history:
      var reactions: seq[string]
      for value in record.reactions:
        reactions.add $value
      var made: seq[string]
      for value in record.foodMade:
        made.add $value
      var charges: seq[string]
      for value in record.charge:
        charges.add $value
      result.add($record.shift & " | " & reactions.join(",") & " | " &
        made.join(",") & " | " & $record.eaten[slot] & " | " &
        $record.coldStarts & " | " & $record.misdrops & " | " &
        charges.join(",") & "\n")

  result.add("\nYOUR NOTES FROM LAST SHIFT:\n" &
    (if cog.notes.len > 0: cog.notes else: "(none)") & "\n\n")
  result.add(operatorBlock(prompt))
  result.add("Reply with ONLY {\"job\":\"" & $jobSupply & "\",\"molecule\":\"" &
    $sim.reactors[0].feed[0] & "\",\"reactor\":\"" & $sim.reactors[0].name &
    "\",\"say\":\"...\",\"notes\":\"...\"} - job one of " & legalJobs() &
    "; molecule one of " & sim.legalMolecules() & "; reactor one of " &
    sim.legalReactors() & "; say at most " & $MaxSayLen &
    " characters; notes at most " & $MaxNotesLen & " characters.")

# ---- Transport ---------------------------------------------------------------

proc extractJsonObject*(text: string): JsonNode =
  ## Parses a raw JSON object or a single standard outer Markdown JSON fence
  ## (` ```json ... ``` ` or ` ``` ... ``` `). Rejects surrounding prose and
  ## non-object JSON values.
  var candidate = text.strip()
  if candidate.startsWith("```"):
    let firstNewline = candidate.find('\n')
    if firstNewline < 0:
      raise newException(ChemistryError,
        "malformed markdown fence in response: " & errorText(candidate))
    let header = candidate[3 ..< firstNewline].strip().toLowerAscii()
    if header.len > 0 and header != "json":
      raise newException(ChemistryError,
        "unsupported markdown fence in response: " & errorText(header))
    let body = candidate[firstNewline + 1 .. ^1]
    if not body.endsWith("```"):
      raise newException(ChemistryError,
        "unclosed markdown fence in response: " & errorText(candidate))
    let inner = body[0 ..< body.len - 3]
    if "```" in inner:
      raise newException(ChemistryError,
        "multiple markdown fences in response")
    candidate = inner.strip()
  if candidate.len < 2 or candidate[0] != '{' or candidate[^1] != '}':
    var head = text.strip()
    if head.len > 160:
      head = head[0 ..< 160] & "..."
    raise newException(ChemistryError,
      "response must be a single JSON object: " & errorText(head))
  try:
    result = parseJson(candidate)
  except CatchableError as error:
    raise newException(ChemistryError, "invalid JSON in response: " & errorText(error.msg))
  if result.kind != JObject:
    raise newException(ChemistryError, "response JSON must be an object")

proc parseDecision*(sim: Sim, payload: JsonNode): Order =
  ## The reply schema in `## Decisions`. An unknown job, a supply without a
  ## reactor, a missing molecule for supply/hoard, an absent species or reactor,
  ## or a non-string field value are all INVALID replies. A feedstock the named
  ## reactor does not take is accepted as written -- the misdrop is the graph
  ## test and must stay expressible.
  if payload.isNil or payload.kind != JObject:
    raise newException(ChemistryError, "decision payload must be a JSON object")
  let sayNode = payload{"say"}
  if not sayNode.isNil and sayNode.kind != JNull:
    if sayNode.kind != JString:
      raise newException(ChemistryError, "say must be a string")
    result.say = sayText(sayNode.getStr())
  let notesNode = payload{"notes"}
  if not notesNode.isNil and notesNode.kind != JNull:
    if notesNode.kind != JString:
      raise newException(ChemistryError, "notes must be a string")
    result.notes = notesText(notesNode.getStr())
  let jobNode = payload{"job"}
  if jobNode.isNil or jobNode.kind != JString:
    raise newException(ChemistryError, "no job in response")
  let job = parseJob(jobNode.getStr())
  if not job.ok:
    raise newException(ChemistryError,
      "unknown job: " & errorText(jobNode.getStr()))
  result.job = job.job
  let moleculeNode = payload{"molecule"}
  if not moleculeNode.isNil and moleculeNode.kind != JNull:
    if moleculeNode.kind != JString:
      raise newException(ChemistryError, "molecule must be a string")
    if moleculeNode.getStr().strip().len > 0:
      let species = parseSpecies(moleculeNode.getStr())
      if not species.ok:
        raise newException(ChemistryError,
          "unknown molecule: " & errorText(moleculeNode.getStr()))
      result.molecule = species.species
      result.hasMolecule = true
  let reactorNode = payload{"reactor"}
  if not reactorNode.isNil and reactorNode.kind != JNull:
    if reactorNode.kind != JString:
      raise newException(ChemistryError, "reactor must be a string")
    if reactorNode.getStr().strip().len > 0:
      let reactor = parseReactorName(reactorNode.getStr())
      if not reactor.ok:
        raise newException(ChemistryError,
          "unknown reactor: " & errorText(reactorNode.getStr()))
      result.reactor = reactor.reactor
      result.hasReactor = true
  let normalized = sim.normalizeOrder(result)
  if not normalized.ok:
    raise newException(ChemistryError, normalized.error)
  result = normalized.order

proc requestFor(client: LlmClient, system, user: string, slot: int):
    tuple[url: string, headers: HttpHeaders, body: string] =
  var body = %*{
    "max_tokens": client.maxOutputTokens,
    "system": system,
    "messages": [{"role": "user", "content": user}]
  }
  var headers: HttpHeaders
  headers["content-type"] = "application/json"
  if client.transport == ltSidecar and slot >= 0:
    headers["X-Coworld-Player-Slot"] = $slot
  if client.transport == ltBedrock:
    body["anthropic_version"] = %BedrockAnthropicVersion
    if client.bedrockToken.len > 0:
      headers["authorization"] = "Bearer " & client.bedrockToken
    result.url = client.bedrockUrl()
  elif client.transport == ltSidecar:
    body["model"] = %client.model
    headers["anthropic-version"] = AnthropicVersion
    result.url = client.sidecarEndpoint & "/v1/messages"
  else:
    body["model"] = %client.model
    ## No `output_config.effort`: Haiku 4.5 rejects the whole request with a
    ## 400 when it is present.
    headers["x-api-key"] = client.apiKey
    headers["anthropic-version"] = AnthropicVersion
    result.url = AnthropicUrl
  result.headers = headers
  result.body = $body

proc isTimeoutError(error: string): bool =
  let lower = error.toLowerAscii()
  "timed out" in lower or "timeout" in lower

proc textOf(client: LlmClient, response: Response, error, url: string): string =
  if error.len > 0:
    if isTimeoutError(error):
      raise newException(ChemistryError, "llm timeout: " & errorText(error))
    raise newException(ChemistryInfraError, "llm transport: " & errorText(error))
  if response.code == 401 or response.code == 403:
    let detail = response.body[0 .. min(response.body.high, 400)]
    if "Model access is denied" in response.body and
        client.tryNextBedrockModel("no model access"):
      raise newException(ChemistryError,
        "bedrock model access denied: " & errorText(detail))
    client.disabled = true
    raise newException(ChemistryInfraError,
      "llm auth failed (" & $response.code & ") at " & url & ": " &
      errorText(detail))
  if response.code == 429:
    let detail = response.body[0 .. min(response.body.high, 300)]
    raise newException(ChemistryInfraError, "llm throttled (429): " &
      errorText(detail))
  if response.code < 200 or response.code >= 300:
    raise newException(ChemistryInfraError, "llm provider error " & $response.code &
      ": " & errorText(response.body[0 .. min(response.body.high, 300)]))
  let payload = parseJson(response.body)
  if payload{"stop_reason"}.getStr() == "refusal":
    raise newException(ChemistryError, "anthropic refusal")
  for contentBlock in payload["content"]:
    if contentBlock{"type"}.getStr() == "text":
      result.add(contentBlock{"text"}.getStr())
  if payload{"stop_reason"}.getStr() == "max_tokens" and '{' notin result:
    raise newException(ChemistryError, "reply cut off at max_tokens before " &
      "any JSON: " & errorText(result[0 .. min(result.high, 160)]))

const RetryHint = "\nYour previous reply was invalid. Respond with ONLY " &
  "the requested JSON object, using one of the listed job, molecule and " &
  "reactor values."

proc decideAll*(
  client: LlmClient,
  sim: Sim,
  prompts: seq[string],
  scripted: seq[ScriptKind],
  skip: seq[bool] = @[]
): seq[SeatDecision] =
  ## One decision result per seat, in slot order. Never raises and never
  ## substitutes a fallback order: a failed or missing model response leaves
  ## `hasOrder = false` so the engine advances without executing an order on
  ## that seat's behalf and records a missed turn.
  result = newSeq[SeatDecision](Seats)
  let capture = getEnv(CogameSaveTrajectoryUriEnv).len > 0
  if capture: client.nativeAttempts = newSeq[seq[DecisionAttempt]](Seats)
  var open: seq[int]
  var lastErrors = newSeq[string](Seats)
  var infraErrors = newSeq[string](Seats)
  for slot in 0 ..< Seats:
    if skip.len == Seats and skip[slot]:
      result[slot] = SeatDecision(hasOrder: false, status: asMissing,
        error: "external or unregistered seat")
      continue
    let kind = scripted[slot]
    if kind != skNone:
      result[slot] = SeatDecision(
        hasOrder: true,
        order: sim.scriptedOrder(slot, kind),
        status: asAccepted)
    elif not sim.cogs[slot].connected:
      result[slot] = SeatDecision(hasOrder: false, status: asMissing,
        error: "seat disconnected")
    elif client.disabled:
      result[slot] = SeatDecision(hasOrder: false, status: asMissing,
        error: "no LLM credentials configured", infraFailure: true)
    else:
      open.add slot
  client.lastBatchSize = open.len
  for attempt in 0 .. 1:
    if open.len == 0 or client.disabled:
      break
    ## ALL open seats in ONE parallel batch -- never sequentially. This is
    ## what keeps 12 shifts inside the 720 s play budget.
    var batch: RequestBatch
    var pendingEvidence: seq[DecisionAttempt]
    for slot in open:
      var user = sim.userPrompt(slot, prompts[slot])
      if attempt > 0:
        user.add(RetryHint)
      let request = client.requestFor(sim.systemPrompt(slot), user, slot)
      if capture:
        pendingEvidence.add(nativeAttempt(sim.policyNames[slot], sim.shift,
          slot, attempt, request.body))
      batch.post(request.url, request.headers, request.body, $slot)
    if attempt == 0:
      client.lastBatchSize = batch.len
    let started = epochTime()
    let responses = client.curl.makeRequests(batch, client.timeoutSeconds)
    let latencyMs = int((epochTime() - started) * 1000.0)
    var stillOpen: seq[int]
    for position, slot in open:
      var evidence: DecisionAttempt
      if capture: evidence = pendingEvidence[position]
      try:
        if capture:
          evidence.receiveNativeResponse(responses[position].response,
            responses[position].error, latencyMs)
        let text = client.textOf(responses[position].response,
          responses[position].error, batch[position].url)
        let payload = extractJsonObject(text)
        var order = sim.parseDecision(payload)
        if capture:
          evidence.parsedAction = order.orderAction()
          evidence.accepted = not order.clamped and
            payload{"say"}.getStr() == order.say and
            payload{"notes"}.getStr() == order.notes
          if not evidence.accepted:
            evidence.rejectionReason = some("native parser clamped or shortened supplied fields")
        order.source = (if attempt == 0: osLlm else: osRetry)
        order.latencyMs = latencyMs
        result[slot] = SeatDecision(
          hasOrder: true,
          order: order,
          status: asAccepted)
      except ChemistryInfraError as error:
        if capture: evidence.rejectionReason = some(error.msg)
        lastErrors[slot] = errorText(error.msg)
        infraErrors[slot] = lastErrors[slot]
        echo "chemistry llm: seat ", slot, " attempt ", attempt, " infra error: ",
          lastErrors[slot]
        stillOpen.add slot
      except CatchableError as error:
        if capture: evidence.rejectionReason = some(error.msg)
        lastErrors[slot] = errorText(error.msg)
        infraErrors[slot] = ""
        echo "chemistry llm: seat ", slot, " attempt ", attempt, " failed: ",
          lastErrors[slot]
        stillOpen.add slot
      if capture: client.nativeAttempts[slot].add(evidence)
    open = stillOpen
  for slot in open:
    let isInfra = infraErrors[slot].len > 0 or client.disabled
    if isInfra:
      echo "chemistry llm: seat ", slot, " infrastructure failure: ",
        lastErrors[slot]
    else:
      echo "chemistry llm: seat ", slot, " missed turn (no fallback): ",
        lastErrors[slot]
    result[slot] = SeatDecision(
      hasOrder: false,
      status: (if lastErrors[slot].len > 0: asRejected else: asMissing),
      error: lastErrors[slot],
      infraFailure: isInfra)
