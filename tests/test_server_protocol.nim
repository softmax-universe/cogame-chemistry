## End-to-end server protocol test against a real local HTTP fixture server:
## 1) Valid HTTP response -> HTTP calls received, real LLM actions applied, ordinary scored results written.
## 2) Provider/transport failure (HTTP 503) -> unscored player_failure.json written, NO ordinary results.json.
## 3) Invalid model reply on shift 1, valid on shift 2 -> real-time missed turn on shift 1, real LLM action on shift 2.

import std/[json, net, os, osproc, streams, strtabs, strutils, tempfiles, unittest]
import chemistry/sim_types

const RepoRoot = currentSourcePath().parentDir().parentDir()
const FixtureScript = RepoRoot / "tools" / "ci" / "fixture_llm_server.py"

proc freePort(): int =
  let sock = newSocket()
  defer: sock.close()
  sock.bindAddr(Port(0), "127.0.0.1")
  int(sock.getLocalAddr()[1])

proc nimPathArgs(): seq[string] =
  if fileExists(RepoRoot / "nim.cfg"):
    return @[]
  for base in ["/tmp/game-actions-a/nimby-pkgs", getHomeDir() / ".nimby" / "pkgs"]:
    if dirExists(base):
      for kind, path in walkDir(base):
        if kind == pcDir:
          if dirExists(path / "src"):
            result.add("--path:" & (path / "src"))
          else:
            result.add("--path:" & path)
      break
  result.add("--path:" & (RepoRoot / "src"))

proc compileBinaries(dir: string): tuple[gameBin, playerBin: string] =
  result.gameBin = dir / "chemistry"
  result.playerBin = dir / "chemistry-player"
  let paths = nimPathArgs()
  for (src, dst, cache) in [
    (RepoRoot / "src" / "chemistry.nim", result.gameBin, dir / "cache-game"),
    (RepoRoot / "src" / "chemistry_player.nim", result.playerBin, dir / "cache-player"),
  ]:
    var args = @["c", "--hints:off", "--warnings:off", "--nimcache:" & cache, "-o:" & dst]
    args.add(paths)
    args.add(src)
    let res = execCmdEx("nim " & quoteShellCommand(args))
    doAssert res.exitCode == 0, "failed to compile " & src & ":\n" & res.output

proc runEpisode(
  gameBin, playerBin, workDir, mode: string
): tuple[exitCode: int, resultsPath, replayPath, failurePath, logPath: string] =
  createDir(workDir)
  let logPath = workDir / "llm_requests.jsonl"
  let fixture = startProcess(
    "python3",
    args = [FixtureScript, "--host", "127.0.0.1", "--mode", mode, "--log-file", logPath],
    options = {poUsePath, poStdErrToStdOut}
  )
  defer:
    fixture.terminate()
    discard fixture.waitForExit(5000)
    fixture.close()
  let fixturePort = fixture.outputStream.readLine().strip()

  let port = freePort()
  let configPath = workDir / "config.json"
  let resultsPath = workDir / "results.json"
  let replayPath = workDir / "replay.json"
  let failurePath = workDir / "player_failure.json"

  var tokens = newJArray()
  var players = newJArray()
  for slot in 0 ..< Seats:
    tokens.add(%("tok-" & $slot))
    players.add(%*{"name": SeatAliases[slot]})
  let cfg = %*{
    "num_agents": Seats,
    "seed": 7,
    "cycles": 3,
    "shifts": 2,
    "ticksPerShift": 20,
    "distractorPeriod": 0,
    "distractorGroundCap": 0,
    "minTurnSeconds": 0,
    "llmTimeoutSeconds": 5,
    "playerConnectTimeoutSeconds": 15,
    "shutdownGraceSeconds": 0,
    "tokens": tokens,
    "players": players
  }
  writeFile(configPath, $cfg)

  let gameEnv = newStringTable(modeCaseSensitive)
  for k, v in envPairs():
    if k notin ["ANTHROPIC_API_KEY", "ANTHROPIC_API_KEY_URI",
                "AWS_ENDPOINT_URL_BEDROCK_RUNTIME", "AWS_BEARER_TOKEN_BEDROCK",
                "COWORLD_LLM_ENDPOINT", "COGAME_SAVE_TRAJECTORY_URI"]:
      gameEnv[k] = v
  gameEnv["COGAME_HOST"] = "127.0.0.1"
  gameEnv["COGAME_PORT"] = $port
  gameEnv["COGAME_CONFIG_URI"] = "file://" & configPath
  gameEnv["COGAME_RESULTS_URI"] = "file://" & resultsPath
  gameEnv["COGAME_SAVE_REPLAY_URI"] = "file://" & replayPath
  gameEnv["COGAME_PLAYER_FAILURE_URI"] = "file://" & failurePath
  gameEnv["COWORLD_LLM_ENDPOINT"] = "http://127.0.0.1:" & fixturePort
  gameEnv["COWORLD_LLM_MODEL"] = "chemistry-fixture"

  let gameProc = startProcess(gameBin, env = gameEnv, options = {poStdErrToStdOut})
  sleep(300)

  var playerProcs: seq[Process]
  let seatScripts = ["", "courier", "courier", "", "courier", "freeloader", "courier", "freeloader"]
  for slot in 0 ..< Seats:
    let pEnv = newStringTable(modeCaseSensitive)
    for k, v in envPairs():
      if k notin ["PLAYER_SCRIPTED", "PLAYER_PROMPT", "COWORLD_PLAYER_WS_URL"]:
        pEnv[k] = v
    pEnv["COWORLD_PLAYER_WS_URL"] =
      "ws://127.0.0.1:" & $port & "/player?slot=" & $slot & "&token=tok-" & $slot
    if seatScripts[slot].len > 0:
      pEnv["PLAYER_SCRIPTED"] = seatScripts[slot]
    playerProcs.add(startProcess(playerBin, env = pEnv, options = {poStdErrToStdOut}))

  let exitCode = gameProc.waitForExit(30000)
  gameProc.close()
  for p in playerProcs:
    discard p.waitForExit(5000)
    p.close()

  (exitCode, resultsPath, replayPath, failurePath, logPath)

suite "real fixture server protocol":
  let workRoot = createTempDir("chemistry-server-proto-", "")
  let (gameBin, playerBin) = compileBinaries(workRoot)

  test "valid HTTP fixture response applies real LLM orders and writes scored results":
    let run = runEpisode(gameBin, playerBin, workRoot / "valid", "valid")
    check run.exitCode == 0
    check fileExists(run.resultsPath)
    check fileExists(run.replayPath)
    check not fileExists(run.failurePath)
    check fileExists(run.logPath)

    let requests = readFile(run.logPath).strip().splitLines()
    check requests.len == 4  # 2 shifts x 2 model seats (slots 0 and 3)
    var seenSlots: seq[string]
    for line in requests:
      let row = parseJson(line)
      check row["path"].getStr() == "/v1/messages"
      check row["status"].getInt() == 200
      seenSlots.add(row["slot"].getStr())
    check "0" in seenSlots
    check "3" in seenSlots

    let results = parseJson(readFile(run.resultsPath))
    check results["reason"].getStr() == "complete"
    check results["missed_turns"][0].getInt() == 0
    check results["missed_turns"][3].getInt() == 0

    let replay = parseJson(readFile(run.replayPath))
    var llmOrders = 0
    for ev in replay["events"]:
      if ev{"k"}.getStr() == "order" and ev{"source"}.getStr() == "llm" and
          ev{"seat"}.getInt() in [0, 3]:
        inc llmOrders
    check llmOrders == 4

  test "provider HTTP 503 outage emits unscored failure and writes NO ordinary results":
    let run = runEpisode(gameBin, playerBin, workRoot / "infra_error", "infra_error")
    check run.exitCode != 0
    check not fileExists(run.resultsPath)
    check fileExists(run.failurePath)
    let failure = parseJson(readFile(run.failurePath))
    check failure["kind"].getStr() == "llm_infrastructure_error"
    check failure["scored"].getBool() == false

  test "invalid model reply causes a real-time missed turn and recovers on next shift":
    let run = runEpisode(gameBin, playerBin, workRoot / "invalid_first_shift", "invalid_first_shift")
    check run.exitCode == 0
    check fileExists(run.resultsPath)
    check not fileExists(run.failurePath)
    let results = parseJson(readFile(run.resultsPath))
    check results["missed_turns"][0].getInt() == 1
    check results["missed_turns"][3].getInt() == 1
    let replay = parseJson(readFile(run.replayPath))
    var missedCount = 0
    var llmOrders = 0
    for ev in replay["events"]:
      if ev{"k"}.getStr() == "missed" and ev{"seat"}.getInt() in [0, 3]:
        inc missedCount
      if ev{"k"}.getStr() == "order" and ev{"source"}.getStr() == "llm" and
          ev{"seat"}.getInt() in [0, 3]:
        inc llmOrders
    check missedCount == 2
    check llmOrders == 2

  removeDir(workRoot)
