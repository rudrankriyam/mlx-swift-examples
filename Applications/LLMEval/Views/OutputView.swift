// Copyright © 2025 Apple Inc.

import MarkdownUI
import SwiftUI

struct OutputView: View {
    let output: String
    let displayStyle: ContentView.DisplayStyle
    let wasTruncated: Bool

    private func processThinkingContent(_ content: String) -> (thinking: String?, after: String?) {
        guard let startRange = content.range(of: "<think>") else {
            return (nil, content.trimmingCharacters(in: .whitespacesAndNewlines))
        }
        guard let endRange = content.range(of: "</think>") else {
            let thinking = String(content[startRange.upperBound...])
                .trimmingCharacters(in: .whitespacesAndNewlines)
            return (thinking, nil)
        }

        let thinking = String(content[startRange.upperBound ..< endRange.lowerBound])
            .trimmingCharacters(in: .whitespacesAndNewlines)
        let afterThink = String(content[endRange.upperBound...])
            .trimmingCharacters(in: .whitespacesAndNewlines)

        return (thinking, afterThink.isEmpty ? nil : afterThink)
    }

    var body: some View {
        ScrollView(.vertical) {
            ScrollViewReader { sp in
                VStack(alignment: .leading, spacing: 12) {
                    let (thinking, afterThink) = processThinkingContent(output)
                    Group {
                        if let thinking {
                            Text(thinking)
                                .italic()
                                .foregroundStyle(.secondary)
                                .textSelection(.enabled)
                        }

                        if let afterThink {
                            if displayStyle == .plain {
                                Text(afterThink)
                                    .textSelection(.enabled)
                            } else {
                                Markdown(afterThink)
                                    .textSelection(.enabled)
                            }
                        } else if thinking == nil {
                            if displayStyle == .plain {
                                Text(output)
                                    .textSelection(.enabled)
                            } else {
                                Markdown(output)
                                    .textSelection(.enabled)
                            }
                        }
                    }

                    // Warning banner when output is truncated
                    if wasTruncated && !output.isEmpty {
                        HStack(spacing: 8) {
                            Image(systemName: "exclamationmark.triangle.fill")
                                .foregroundStyle(.orange)
                            Text("Output truncated: Maximum token limit reached")
                                .font(.caption)
                                .foregroundStyle(.secondary)
                        }
                        .padding(8)
                        .background(.orange.opacity(0.1), in: RoundedRectangle(cornerRadius: 6))
                    }
                }
                .onChange(of: output) { _, _ in
                    sp.scrollTo("bottom")
                }

                Spacer()
                    .frame(width: 1, height: 1)
                    .id("bottom")
            }
        }
    }
}
