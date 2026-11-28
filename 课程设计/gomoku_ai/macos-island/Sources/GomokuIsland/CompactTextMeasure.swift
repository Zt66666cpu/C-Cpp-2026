import AppKit

/// 使用与 CompactViews 相同的等宽数字系统字体测量胶囊文字真实宽度。
enum CompactTextMeasure {
    private static let font = NSFont.monospacedDigitSystemFont(ofSize: 12, weight: .medium)

    static func fullText(for state: MonitorCompactState) -> String {
        switch state {
        case .training(let round, _, _, let heurPct, _):
            if let heurPct {
                return "R\(round) \(heurPct)%"
            }
            return "R\(round) 训练"
        case .paused(let round, let heurPct):
            if round > 0 {
                if let heurPct {
                    return "R\(round) \(heurPct)%"
                }
                return "R\(round) 暂停"
            }
            return "待机"
        case .offline:
            return "离线"
        case .hidden:
            return ""
        }
    }

    static func fullTextWidth(for state: MonitorCompactState) -> CGFloat {
        let text = fullText(for: state)
        guard !text.isEmpty else { return 0 }
        return ceil((text as NSString).size(withAttributes: [.font: font]).width) + 1
    }
}
