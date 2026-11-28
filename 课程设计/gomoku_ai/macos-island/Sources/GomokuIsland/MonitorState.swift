import Foundation

enum MonitorCompactState: Equatable {
    case training(round: Int, maxRounds: Int, stageLabel: String, heurPct: Int?, progress: Double)
    case paused(round: Int, heurPct: Int?)
    case offline
    case hidden
}

@MainActor
final class MonitorPoller: ObservableObject {
    static let hideWhenIdleKey = "gomokuHideWhenIdle"
    static let selectedRunDefaultsKey = "gomokuSelectedRunKey"

    @Published private(set) var compactState: MonitorCompactState = .offline
    @Published private(set) var selectedRunKey: String?
    @Published private(set) var latestSnapshotJSON: String?

    var monitorBaseURL: String = "http://127.0.0.1:8766"
    var onCompactStateChanged: ((MonitorCompactState) -> Void)?
    var onSnapshotUpdated: ((String) -> Void)?
    var onPromptRequested: ((String, String) -> Void)?

    var hideWhenIdle: Bool {
        get { UserDefaults.standard.bool(forKey: Self.hideWhenIdleKey) }
        set {
            UserDefaults.standard.set(newValue, forKey: Self.hideWhenIdleKey)
            recomputeCompactFromLast()
        }
    }

    private var timer: Timer?
    private var inFlight = false
    private var lastRawStatus: [String: Any]?
    private var lastPromotedRoundByRun: [String: Int] = [:]
    private var initializedPromotedRuns = Set<String>()

    init() {
        selectedRunKey = UserDefaults.standard.string(forKey: Self.selectedRunDefaultsKey)
    }

    func start() {
        stop()
        pollNow()
        let t = Timer(timeInterval: 3.0, repeats: true) { [weak self] _ in
            Task { @MainActor in
                self?.pollNow()
            }
        }
        RunLoop.main.add(t, forMode: .common)
        timer = t
    }

    func stop() {
        timer?.invalidate()
        timer = nil
    }

    func selectRun(_ key: String?) {
        let trimmed = key?.trimmingCharacters(in: .whitespacesAndNewlines)
        selectedRunKey = (trimmed?.isEmpty == false) ? trimmed : nil
        if let selectedRunKey {
            UserDefaults.standard.set(selectedRunKey, forKey: Self.selectedRunDefaultsKey)
        } else {
            UserDefaults.standard.removeObject(forKey: Self.selectedRunDefaultsKey)
        }
        pollNow()
    }

    func pollNow() {
        guard !inFlight else { return }
        guard var comps = URLComponents(string: "\(monitorBaseURL)/api/status") else { return }
        if let selectedRunKey, !selectedRunKey.isEmpty {
            comps.queryItems = [URLQueryItem(name: "run", value: selectedRunKey)]
        }
        guard let url = comps.url else { return }

        inFlight = true
        var req = URLRequest(url: url)
        req.timeoutInterval = 3.5

        URLSession.shared.dataTask(with: req) { [weak self] data, _, error in
            let payloadStr = data.flatMap { String(data: $0, encoding: .utf8) }
            Task { @MainActor [weak self] in
                guard let self else { return }
                self.inFlight = false
                guard error == nil,
                      let data,
                      let payloadStr,
                      let json = (try? JSONSerialization.jsonObject(with: data)) as? [String: Any] else {
                    self.applyOfflineState()
                    return
                }
                self.handleFetchedStatus(json: json, rawString: payloadStr)
            }
        }.resume()
    }

    private func handleFetchedStatus(json: [String: Any], rawString: String) {
        if let preferred = chooseDefaultRunIfNeeded(from: json) {
            if preferred != selectedRunKey {
                selectRun(preferred)
                return
            }
        }

        lastRawStatus = json
        latestSnapshotJSON = rawString
        onSnapshotUpdated?(rawString)

        if let selected = json["selected"] as? [String: Any] {
            let rKey = (selected["key"] as? String) ?? (selected["run"] as? String) ?? ""
            let isRunning = (selected["running"] as? Bool) ?? false
            let gomoku = selected["gomoku"] as? [String: Any]
            let latest = selected["latest"] as? [String: Any]
            let completedRound = (gomoku?["completed_rounds"] as? Int)
                ?? (latest?["round"] as? Int)
                ?? 0
            let promoted = (latest?["promoted"] as? Bool) ?? false
            if !rKey.isEmpty {
                let prev = lastPromotedRoundByRun[rKey]
                if !initializedPromotedRuns.contains(rKey) {
                    initializedPromotedRuns.insert(rKey)
                    lastPromotedRoundByRun[rKey] = promoted ? completedRound : 0
                } else if isRunning && promoted && completedRound > (prev ?? 0) {
                    lastPromotedRoundByRun[rKey] = completedRound
                    if let heur = latestHeurPct(selected: selected) {
                        onPromptRequested?("🎉 R\(completedRound) 晋级冠军 · Heur \(heur)%", "summary")
                    } else {
                        onPromptRequested?("🎉 R\(completedRound) 晋级新冠军！", "summary")
                    }
                }
            }
        }

        recomputeCompactFromLast()
    }

    private func chooseDefaultRunIfNeeded(from json: [String: Any]) -> String? {
        guard let runs = json["runs"] as? [[String: Any]], !runs.isEmpty else { return nil }
        let keys = Set(runs.compactMap { ($0["key"] as? String) ?? ($0["run"] as? String) })
        if let cur = selectedRunKey, keys.contains(cur) {
            // If current selection is idle/non-gomoku and a Gomoku run is actively running, stay only if cur is Gomoku or explicitly paused? Keep explicit user selection once valid.
            return nil
        }

        let gomokuRuns = runs.filter { row in
            let k = ((row["key"] as? String) ?? (row["run"] as? String) ?? "").lowercased()
            return k.contains("gomoku") || row["gomoku"] != nil
        }
        if let active = gomokuRuns.first(where: { ($0["running"] as? Bool) == true }),
           let k = (active["key"] as? String) ?? (active["run"] as? String) {
            return k
        }
        for preferred in ["windows-4070/gomoku-v4-evolve", "windows-4070/gomoku-v3-pnnue", "gomoku-v2"] {
            if keys.contains(preferred) {
                return preferred
            }
        }
        if let firstGomoku = gomokuRuns.first,
           let k = (firstGomoku["key"] as? String) ?? (firstGomoku["run"] as? String) {
            return k
        }
        return nil
    }

    private func recomputeCompactFromLast() {
        guard let json = lastRawStatus else {
            applyOfflineState()
            return
        }
        guard let selected = json["selected"] as? [String: Any] else {
            updateCompactState(hideWhenIdle ? .hidden : .paused(round: 0, heurPct: nil))
            return
        }

        let isRunning = (selected["running"] as? Bool) ?? false
        let gomoku = selected["gomoku"] as? [String: Any]
        let latest = selected["latest"] as? [String: Any]
        let completedRounds = (gomoku?["completed_rounds"] as? Int)
            ?? (latest?["round"] as? Int)
            ?? (latest?["iter"] as? Int)
            ?? 0
        let currentRound = (gomoku?["current_round"] as? Int) ?? max(1, completedRounds)
        let maxRounds = (gomoku?["max_rounds"] as? Int) ?? (selected["max_iters"] as? Int) ?? 0
        let stageLabel = (gomoku?["stage_label"] as? String) ?? "训练中"
        let progress = (selected["progress"] as? Double) ?? 0.5
        let heurPct = latestHeurPct(selected: selected)

        if isRunning {
            updateCompactState(
                .training(
                    round: currentRound,
                    maxRounds: maxRounds,
                    stageLabel: stageLabel,
                    heurPct: heurPct,
                    progress: min(1.0, max(0.04, progress > 0 ? progress : 0.35))
                )
            )
        } else if hideWhenIdle {
            updateCompactState(.hidden)
        } else {
            updateCompactState(.paused(round: completedRounds, heurPct: heurPct))
        }
    }

    private func latestHeurPct(selected: [String: Any]) -> Int? {
        if let g = selected["gomoku"] as? [String: Any],
           let s = g["latest_heuristic_score"] as? Double {
            return Int((s * 100.0).rounded())
        }
        if let latest = selected["latest"] as? [String: Any],
           let vh = latest["vs_heuristic"] as? [String: Any],
           let s = vh["score"] as? Double {
            return Int((s * 100.0).rounded())
        }
        return nil
    }

    private func applyOfflineState() {
        lastRawStatus = nil
        updateCompactState(hideWhenIdle ? .hidden : .offline)
    }

    private func updateCompactState(_ next: MonitorCompactState) {
        guard next != compactState else { return }
        compactState = next
        onCompactStateChanged?(next)
    }
}
